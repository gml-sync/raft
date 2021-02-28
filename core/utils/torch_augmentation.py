import torch

def coords_grid(batch, ht, wd): # copied from utils
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def check_out_of_bound(flow, occ):
    """ Add new occlusions for cropping/zoom """
    
    _c, height, width = flow.size()
    u = flow[:1, :, :] # b (c=1) h w
    v = flow[1:, :, :]
    grid = coords_grid(1, height, width)
    xx = grid[0, :1] # b (c=1) h w
    yy = grid[0, 1:]
    xx = xx + u
    yy = yy + v

    out_of_bound = ((xx < 0) | (yy < 0) | (xx >= width) | (yy >= height)).float()
    occ = torch.clamp(out_of_bound + occ, 0, 1)

    return occ