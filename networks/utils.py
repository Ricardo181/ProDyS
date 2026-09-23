import cv2
import numpy as np


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        print("--- Initializing ActivationsAndGradients ---")
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            print(f"Attempting to register hooks for layer: {target_layer.__class__.__name__}")
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        print(f"Hook: save_activation called for {module.__class__.__name__}")
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        print(f"Hook: Appending activation with shape: {activation.shape}")
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        print(f"Hook: save_gradient called for {module.__class__.__name__}")
        grad = None
        if grad_output is not None and grad_output and grad_output[0] is not None:
            grad = grad_output[0]
            print(f"Hook: Original grad_output[0] shape: {grad.shape}")
        else:
            print(f"Hook: grad_output is None or empty or grad_output[0] is None: {grad_output}")

        if grad is not None:
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
                print(f"Hook: Reshaped gradient shape: {grad.shape}")
            print(f"Hook: Prepending gradient with shape: {grad.cpu().detach().shape}")
            self.gradients = [grad.cpu().detach()] + self.gradients
        else:
            print("Hook: No valid gradient to prepend.")

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        if grads.ndim == 4:
            # 处理 CNN 类型的4D梯度 (B, C, H, W)
            # 有些实现中通道可能在axis=1，梯度是 (B,C,H,W)，所以这里 axis=(2,3) 是对 H,W 求平均
            return np.mean(grads, axis=(2, 3), keepdims=True)
        elif grads.ndim == 3:
            # 处理 Transformer 类型的3D梯度 (B, L, C)
            # L 是序列长度 (例如 784 patch), C 是通道数 (例如 192)
            # 我们需要对 L (axis=1) 求平均，得到每个通道 C 的权重
            # 返回的权重形状将是 (B, 1, C)
            return np.mean(grads, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unsupported grads dimension: {grads.ndim}. Expected 3 or 4.")

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]].sum()
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        # 正向传播得到网络输出logits(未经过softmax)
        output = self.activations_and_grads(input_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


# def show_cam_on_image(img: np.ndarray,
#                       mask: np.ndarray,
#                       use_rgb: bool = False,
#                       colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
#     """ This function overlays the cam mask on the image as an heatmap.
#     By default the heatmap is in BGR format.

#     :param img: The base image in RGB or BGR format.
#     :param mask: The cam mask.
#     :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
#     :param colormap: The OpenCV colormap to be used.
#     :returns: The default image with the cam overlay.
#     """

#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
#     if use_rgb:
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#     heatmap = np.float32(heatmap) / 255

#     if np.max(img) > 1:
#         raise Exception(
#             "The input image should np.float32 in the range [0, 1]")

#     cam = heatmap + img
#     cam = cam / np.max(cam)
#     return np.uint8(255 * cam)

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    This function overlays the cam mask on the image as a heatmap.
    By default, the heatmap is in BGR format.

    :param img: The base image in grayscale or RGB format.
    :param mask: The cam mask (single-channel heatmap).
    :param use_rgb: Whether to use an RGB or BGR heatmap (set True if 'img' is in RGB format).
    :param colormap: The OpenCV colormap to be used.
    :returns: The resulting image with the heatmap overlay.
    """
    # print("mask: ",mask.min(), mask.max())

    # Ensure mask is in range [0, 1]
    mask = np.clip(mask, 0, 1)

    # Apply colormap to the mask
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)

    # Convert heatmap to RGB if necessary
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    heatmap = np.float32(heatmap) / 255  # Normalize heatmap to [0, 1]

    # print("img: ",img.min(), img.max())
    # Normalize the input image to [0, 1] and expand to 3 channels if grayscale
    img = img.astype(np.float32)
    # print("img_after: ",img.min(), img.max())
    if len(img.shape) == 2:  # If grayscale
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)  # Normalize
        img = np.stack([img, img, img], axis=-1)  # Convert to 3-channel

    if np.max(img) > 1:
        raise Exception("The input image should be np.float32 in the range [0, 1]")

    # Combine heatmap and original image
    cam = 0.5 * heatmap + 0.5 * img
    cam = cam / np.max(cam)  # Normalize the result to [0, 1]
    cv2.imwrite("heatmap.jpg", heatmap * 255)  # 保存伪彩色热力图
    cv2.imwrite("normalized_image.jpg", img * 255)  # 保存归一化后的灰度图
    cv2.imwrite("combined_cam.jpg", cam * 255)  # 保存叠加结果

    return np.uint8(255 * cam)


def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h+size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w+size]

    return img

def reshape_transform(tensor):
    # tensor: (B, L, C)
    B, L, C = tensor.shape
    H_W_patch = int(np.sqrt(L))
    assert H_W_patch * H_W_patch == L, "Patch数不是完全平方数"
    result = tensor.reshape(B, H_W_patch, H_W_patch, C)
    result = result.permute(0, 3, 1, 2)  # (B, C, H, W)
    return result
