import torch
import MNN

class export_Model:
    def __init__(self, model, file_name, save_pth=0):
        """
        Args:
            model: Torch.nn
            file_name: export filename
            save_pth: E.g 1? 0  
        """
        self.model = model
        if file_name.spilt('.')[1] != ".pth":
            self.file_name = file_name.spilt('.')[0]+".pth"
        self.file_name = file_name
        self.model_type = "pth"
        if save_pth==1:torch.save(model, file_name)

    def onnx(self,dummy_input, inp_name, oup_name,opset_version=11):
        """
        pip install onnx
        pip install onnxruntime
        
        Turn pth to onnx

        Args:
            inp_name: model input name
            oup_name: model output name

        Examples:
        .. code-block:: python
            model = NNet()
            dummy_input = torch.randn(1, 3, 125, 125)
            export_Model(model,"nnet.pth").onnx(dummy_input, "input_name", "output_name")
        """
        self.dummy_input = dummy_input
        self.input_name = inp_name
        self.oup_name = oup_name
        self.opset_version = opset_version

        self.model.load_state_dict(torch.load(self.file_name, map_location='cpu'))
        self.model.eval()
        self.file_name = self.file_name.spilt(".")[0]+"onnx"
        torch.onnx.export(
            self.model,
            self.dummy_input,
            self.file_name,
            input_names=[self.input_name],
            output_names=[self.oup_name],
            dynamic_axes={self.input_name: {0: "batch_size"}, self.oup_name: {0: "batch_size"}},
            opset_version=self.opset_version
        )

        self.model_type = "ONNX"

    def tflittle(self):

        self.model_type = "TFLITE"

    def tf(self):

        self.model_type = "TF"

    def turn_mnn(self,fp16=0 ,bizcode="mobilenet", optim=0):
        """
        MNN Doc
            https://mnn-docs.readthedocs.io/en/latest/tools/convert.html

        pip install -U mnn
        Args:
            fp16: 将conv/matmul/LSTM的float32参数保存为float16
            bizcode: MNN
            optim:  图优化选项,默认为0:
                        - 0:正常优化
                        - 1:优化后模型尽可能小
                        - 2:优化后模型尽可能快
        """
        fp="--fp16" if fp16==1 else fp=""
        __command = "mnnconvert -f {model_type} --modelFile {model_path} --MNNModel {model_export_name} --bizCode {bizcode} {fp}  --optimizePrefer {optim}".format(
            model_type=self.model_type,
            model_path=self.file_name,
            model_export_name=self.file_name.spilt(".")[0]+".mnn",
            bizcode = self.bizcode,
            fp=fp,
            optim = self.optim
        )
