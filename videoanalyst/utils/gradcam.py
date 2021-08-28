import numpy as np
import cv2


class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, ):
        self.feature = None
        self.gradient = None
        self.handlers = []

    def features_hook(self, module, input, output):
        self.feature = output
        print("feature shape:{}".format(output.size()))

    def grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient = output_grad[1].permute(1, 2, 0).resize(2, 256, 25, 25)
        # print("feature shape:{}".format(len(output_grad)))

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self.features_hook))
                self.handlers.append(module.register_backward_hook(self.grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self):
        """
        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """

        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        # 数值归一化
        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to 224*224
        cam = cv2.resize(cam, (127, 127))
        return cam