import math
import torch


def gauss_kernel(device=torch.device('cpu'), channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel


def downsample(x):
    return x[:, :, ::2, ::2]


def downsample_antialias(x, kernel):
    x = conv_gauss(x, kernel)
    return downsample(x)


def downsample_antialias_n_times(x, kernel, n):
    for _ in range(n):
        x = downsample_antialias(x, kernel)
    return x


def downsample_n_times_upsample_n_times(x, n):
    for _ in range(n):
        x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
    for _ in range(n):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
    return x


def blur_n_times(x, kernel, n):
    for _ in range(n):
        x = conv_gauss(x, kernel)
    return x


def downsample_n_times(x, n):
    for _ in range(n):
        x = downsample(x)
    return x


def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels=x.shape[1], device=x.device))


def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out


def create_gaussian_pyramid(img, kernel, max_levels):
    gaussian_pyramid = [img]
    for i in range(1, max_levels):
        filtered = conv_gauss(gaussian_pyramid[i - 1], kernel)
        down = downsample(filtered)
        gaussian_pyramid.append(down)
    return gaussian_pyramid[::-1]


def gaussian_to_laplacian_pyramid(gaussian_pyramid):
    tmp_gaussian_pyramid = gaussian_pyramid[::-1]
    laplacian_pyramid = []
    for i in range(len(tmp_gaussian_pyramid) - 1):
        upsampled = upsample(tmp_gaussian_pyramid[i + 1])
        laplacian = tmp_gaussian_pyramid[i] - upsampled
        laplacian_pyramid.append(laplacian)

    laplacian_pyramid.append(tmp_gaussian_pyramid[-1])
    return laplacian_pyramid[::-1]


def laplacian_to_gaussian_pyramid(laplacian_pyramid):
    gaussian_pyramid = [laplacian_pyramid[0]]
    for i in range(1, len(laplacian_pyramid)):
        upsampled = upsample(gaussian_pyramid[i - 1])
        gaussian_pyramid.append(upsampled + laplacian_pyramid[i])
    return gaussian_pyramid


def is_powerof_two(x):
    return bool(x) and (not (x & (x - 1)))


def find_next_power_of_two(x):
    n = math.ceil(math.log(x) / math.log(2))
    return 2 ** n


def pad_to_next_power_of_two(img):
    _, _, H, W = img.shape
    new_H = find_next_power_of_two(H)
    new_W = find_next_power_of_two(W)
    return torch.nn.functional.pad(img, (0, new_W - W, 0, new_H - H), mode='reflect')
