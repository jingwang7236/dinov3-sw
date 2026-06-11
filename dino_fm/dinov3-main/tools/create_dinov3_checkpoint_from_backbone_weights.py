from argparse import ArgumentParser
import torch 
import argparse

# parser = ArgumentParser()
# parser.add_argument('weights_path', help='Path to the weights of the pretrained backbone')
# # parser.add_argument('output_path', help='Path to save the output checkpoint file (optional)',  default=None)

# args = parser.parse_args()
# args.output_path = args.weights_path.replace('.pth', '_dinov3_checkpoint.pth')
# # args.output_path = args.output_path or args.weights_path.replace('.pth', '_dinov3_checkpoint.pth')

# sd = torch.load(args.weights_path, map_location='cpu')
# sd = {
#     f'backbone.{key}': val for key, val in sd.items()
# }
# sd = {
#     "teacher": sd
# }
# torch.save(sd, args.output_path)




def coerce_to_dense_cpu(x):
    """把可能的 DTensor/CUDA Tensor 统一变成 CPU 上的普通 torch.Tensor。"""
    # 1) DTensor -> Replicate -> to_local()
    try:
        from torch.distributed.tensor import DTensor, Replicate
        if isinstance(x, DTensor):
            # 用它自己的 mesh 在 Replicate 上还原完整权重
            x = x.redistribute(device_mesh=x.device_mesh, placements=[Replicate()]).to_local()
    except Exception:
        pass
    # 2) 变成普通 Tensor 并放到 CPU（去掉设备依赖）
    if isinstance(x, torch.Tensor):
        if x.is_sparse:
            x = x.coalesce()
        x = x.detach().clone().cpu()
        return x
    # 3) 其它可转对象尽量 as_tensor 到 CPU
    try:
        return torch.as_tensor(x).cpu()
    except Exception:
        return x  # 转不了的直接返回，下面会跳过非 tensor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights_path", help="path to pretrained backbone weights")
    ap.add_argument("--output_path", default=None, help="path to save converted checkpoint")
    ap.add_argument("--wrap_with_backbone", action="store_true",
                    help="prefix keys with 'backbone.' (for DINOv3 teacher ckpt format)")
    args = ap.parse_args()
    out_path = args.output_path or args.weights_path.replace(".pth", "_dinov3_checkpoint_new.pth")

    # 一定要 map 到 CPU，避免把 GPU/DTensor 也带进来
    raw = torch.load(args.weights_path, map_location="cpu")

    # 兼容三种常见结构：state_dict 本体 / {"state_dict": ...} / {"teacher": ...}
    if isinstance(raw, dict) and "teacher" in raw:
        sd = raw["teacher"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
    else:
        sd = raw

    converted = {}
    for k, v in sd.items():
        vv = coerce_to_dense_cpu(v)
        if not isinstance(vv, torch.Tensor):
            # 忽略非张量条目（比如一些元数据）
            continue
        new_k = f"backbone.{k}" if args.wrap_with_backbone and not k.startswith("backbone.") else k
        converted[new_k] = vv

    final = {"teacher": converted}

    # 保存前做一次自检：不得包含 DTensor/非CPU
    try:
        from torch.distributed.tensor import DTensor
        assert all(not isinstance(t, DTensor) for t in final["teacher"].values())
    except Exception:
        pass
    assert all(isinstance(t, torch.Tensor) and (not t.is_cuda) for t in final["teacher"].values()), \
        "All tensors must be CPU dense tensors before saving."

    torch.save(final, out_path)
    print(f"[ok] saved to {out_path}, params={len(final['teacher'])}")

if __name__ == "__main__":
    main()