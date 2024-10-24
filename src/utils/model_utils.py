import torch


def generate_pseudo_masks(ts_size: int, num_samples: int):

    # Tạo mask với toàn bộ giá trị False
    masks = torch.zeros((num_samples, ts_size), dtype=torch.bool)

    return masks


def generate_random_masks(num_samples: int, ts_size: int, mask_size: int, num_masks: int):
    num_patches = int(ts_size // mask_size)  # Số lượng patch

    def single_sample_mask():
        idx = torch.randperm(num_patches)[:num_masks]  # Lựa chọn ngẫu nhiên các patch
        mask = torch.zeros(ts_size, dtype=torch.bool)
        for j in idx:
            mask[j * mask_size:(j + 1) * mask_size] = 1  # Gán mask vào các patch
        return mask

    # Tạo danh sách mask cho mỗi sample
    masks_list = [single_sample_mask() for _ in range(num_samples)]
    
    # Stack các mask thành tensor với shape (num_samples, ts_size)
    masks = torch.stack(masks_list, dim=0)  
    
    return masks


def mask_it(x: torch.Tensor, masks: torch.Tensor):
    """
    Args:
        x.shape = (batch_size, ts_size, f)
        masks.shape = (batch_size, ts_size), với mỗi giá trị là True hoặc False (True nghĩa là bị mask)

    """

    b, l, f = x.shape  # b: batch_size, l: ts_size, f: f
    
    # Đảm bảo masks có shape là (batch_size, ts_size)
    assert masks.shape == (b, l), "Shape của masks phải là (batch_size, ts_size)"
    
    # Mở rộng mask sang f (feature dimension) để khớp với x
    masks_expanded = masks.unsqueeze(-1).expand(-1, -1, f)  # (batch_size, ts_size, f)
    
    # Chỉ giữ lại các phần tử không bị mask (masks == 0)
    x_visible = x[~masks_expanded].reshape(b, -1, f)  # (batch_size, vis_size, f)
    
    return x_visible