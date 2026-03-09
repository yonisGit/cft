import cv2
import torch
import numpy as np


def get_interpolated_values(baseline, target, num_steps):
    """Returns a tensor of all image interpolation steps."""
    if num_steps <= 0:
        return torch.tensor([])
    if num_steps == 1:
        return torch.stack([baseline, target])

    delta = target - baseline
    scales = torch.linspace(0, 1, num_steps + 1, dtype=torch.float32, device=baseline.device)

    # Dynamically reshape scales to broadcast correctly based on baseline dimensions
    target_shape = (num_steps + 1,) + (1,) * baseline.ndim
    scales = scales.view(target_shape)

    deltas = scales * delta.unsqueeze(0)
    interpolated_activations = baseline.unsqueeze(0) + deltas

    return interpolated_activations


def _compute_target_score(output, target_index):
    """Helper to compute the one-hot target score for backpropagation."""
    batch_size = output.size(0)

    if target_index is None:
        target_index = output.argmax(dim=-1)
    elif not isinstance(target_index, torch.Tensor):
        target_index = torch.tensor(target_index, device=output.device)

    if target_index.ndim == 0:
        target_index = target_index.expand(batch_size)

    with torch.no_grad():
        mask = torch.zeros_like(output)
        mask.scatter_(1, target_index.unsqueeze(1).to(output.device), 1.0)

    one_hot = mask.clone().requires_grad_(True)
    target_score = torch.sum(one_hot * output)
    return target_score


def avg_heads_iia(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    return cam.clamp(min=0)


def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-3], cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, cam.shape[-3], grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    return cam.clamp(min=0).mean(dim=1)


def apply_self_attention_rules(r_ss, cam_ss):
    return torch.matmul(cam_ss, r_ss)


def upscale_relevance(relevance):
    relevance = relevance.reshape(-1, 1, 14, 14)
    relevance = torch.nn.functional.interpolate(relevance, scale_factor=16, mode='bilinear')
    relevance = relevance.reshape(relevance.shape[0], -1)

    rel_min = relevance.min(dim=1, keepdim=True)[0]
    rel_max = relevance.max(dim=1, keepdim=True)[0]
    relevance = (relevance - rel_min) / (rel_max - rel_min)

    return relevance.reshape(-1, 1, 224, 224)


def generate_relevance(model, input_tensor, method='lrp', index=None):
    if method == 'iia':
        relevance = get_iia_vit(model, input_tensor, index)
    elif method == 'gradcam':
        relevance = get_gradcam_vit(index, input_tensor, model)
    elif method == 'lrp':
        relevance = lrp(index, input_tensor, model)
    else:
        raise ValueError(f"Unknown method: {method}")

    return upscale_relevance(relevance)


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    return cam / np.max(cam)


def get_image_with_relevance(image, relevance):
    image = image.permute(1, 2, 0)
    relevance = relevance.permute(1, 2, 0)

    image = (image - image.min()) / (image.max() - image.min())
    image = 255 * image
    vis = image * relevance
    return vis.data.cpu().numpy()


def get_gradcam_vit(index, input_tensor, model):
    output = model(input_tensor, register_hook=True)
    target_score = _compute_target_score(output, index)

    model.zero_grad()
    cams = []

    for blk in model.blocks:
        grad = torch.autograd.grad(target_score, [blk.attn.attention_map], retain_graph=True)[0]
        cam = blk.attn.get_attn_cam()

        cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])

        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        cams.append(cam.unsqueeze(0))

    relevance = compute_rollout_attention(cams)[:, 0, 1:]
    return relevance


def get_iia_vit(vit_ours_model, image, index):
    aggregate_before = True
    device = image.device

    target_image = image.squeeze().cpu()
    baseline_image = torch.zeros_like(target_image)
    imgs = get_interpolated_values(baseline_image, target_image, num_steps=10)

    attention_probs = []
    attention_grads = []

    for interp_image in imgs:
        interp_image = interp_image.unsqueeze(0).cuda()
        output = vit_ours_model(interp_image, register_hook=True)

        target_score = _compute_target_score(output, index)
        vit_ours_model.zero_grad()
        target_score.backward(retain_graph=True)

        cams = []
        block_attn = []

        for blk in vit_ours_model.blocks:
            grad = blk.attn.get_attn_gradients()
            cam = blk.attn.get_attention_map()
            block_attn.append(cam)
            cam = avg_heads_iia(cam, grad)
            cams.append(cam)

        attn = torch.stack(block_attn)
        baseline_attn = torch.zeros_like(attn.cpu())
        iiattn = get_interpolated_values(baseline_attn, attn.cpu(), num_steps=10)

        cams = []
        grads = []

        for iatn in iiattn:
            iatn.requires_grad = True
            output = vit_ours_model(
                interp_image,
                register_hook=True,
                attn_prob=iatn[-1].cuda()
            )

            target_score = _compute_target_score(output, index)
            vit_ours_model.zero_grad()
            target_score.backward(retain_graph=True)

            with torch.no_grad():
                block_cams = []
                block_grads = []

                for blk in vit_ours_model.blocks:
                    grad = blk.attn.get_attn_gradients()
                    cam = blk.attn.get_attention_map()

                    if aggregate_before:
                        cam = avg_heads_iia(cam, grad)
                    else:
                        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
                        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])

                    block_cams.append(cam)
                    block_grads.append(grad)

            cams.append(torch.stack(block_cams))
            grads.append(torch.stack(block_grads))

        attention_probs.append(torch.stack(cams))
        attention_grads.append(torch.stack(grads))

    integrated_attention = torch.stack(attention_probs)
    integrated_gradients = torch.stack(attention_grads)

    final_attn = torch.mean(integrated_attention, dim=[0, 1])
    final_grads = torch.mean(integrated_gradients, dim=[0, 1])

    final_rollout = rollout(
        attentions=final_attn.unsqueeze(1),
        head_fusion="mean",
        gradients=None if aggregate_before else final_grads.unsqueeze(1)
    )

    return final_rollout


def lrp(index, input_tensor, model):
    batch_size = input_tensor.size(0)
    output = model(input_tensor, register_hook=True)
    target_score = _compute_target_score(output, index)

    model.zero_grad()
    num_tokens = model.blocks[0].attn.get_attention_map().shape[-1]

    r_matrix = torch.eye(num_tokens, device='cuda')
    r_matrix = r_matrix.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)

    for blk in model.blocks:
        grad = torch.autograd.grad(target_score, [blk.attn.attention_map], retain_graph=True)[0]
        cam = blk.attn.get_attention_map()
        cam = avg_heads(cam, grad)
        r_matrix = r_matrix + apply_self_attention_rules(r_matrix, cam)

    relevance = r_matrix[:, 0, 1:]
    return relevance


def rollout(
        attentions,
        discard_ratio: float = 0,
        start_layer=0,
        head_fusion="max",
        gradients=None,
        return_resized: bool = True,
):
    all_layer_attentions = []
    attn = []

    if gradients is not None:
        for attn_score, grad in zip(attentions, gradients):
            score_grad = attn_score * grad
            attn.append(score_grad.clamp(min=0).detach())
    else:
        attn = attentions

    for attn_heads in attn:
        if head_fusion == "mean":
            fused_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
        elif head_fusion == "max":
            fused_heads = (attn_heads.max(dim=1)[0]).detach()
        elif head_fusion == "sum":
            fused_heads = (attn_heads.sum(dim=1)[0]).detach()
        elif head_fusion == "median":
            fused_heads = (attn_heads.median(dim=1)[0]).detach()
        elif head_fusion == "min":
            fused_heads = (attn_heads.min(dim=1)[0]).detach()

        flat = fused_heads.view(fused_heads.size(0), -1)
        _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)

        flat[0, indices] = 0
        all_layer_attentions.append(fused_heads)

    rollout_arr = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
    mask = rollout_arr[:, 0, 1:]

    if return_resized:
        width = int(mask.size(-1) ** 0.5)
        mask = mask.reshape(width, width)

    return mask


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    device = all_layer_matrices[0].device

    eye = torch.eye(num_tokens, device=device).expand(batch_size, num_tokens, num_tokens)
    all_layer_matrices = [matrix + eye for matrix in all_layer_matrices]

    matrices_aug = [
        matrix / matrix.sum(dim=-1, keepdim=True)
        for matrix in all_layer_matrices
    ]

    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer + 1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)

    return joint_attention