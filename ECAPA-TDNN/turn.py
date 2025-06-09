import torch
import time
import json
import logging.config
import os
import onnx

class Config:
    """
    Config example :
    time logging train device ...
    """
    def __init__(self):
        self.__start_time = time.time()
        self.hours = 0
        self.minute = 0
        self.second = 0
        self.min_memory=1024*1024

    def get_spend_time(self):
        """ 
        get spend time (hours, minutes, sec)
        """
        self.second = int(time.time()-self.__start_time)+1
        
        if self.second > 60:
            self.minute = self.second/60
            self.second = self.second-(self.minute*60)
            if self.minute>60:
                self.hours = self.minute/60
                self.minute = self.minute-(self.hours*60)

        return (self.hours,self.minute,self.second)

    def device(self):
        """
        select free device
        """
        if torch.cuda.is_available():
            selected_gpu = 0
            max_free_memory = -1
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free_memory = props.total_memory - torch.cuda.memory_reserved(i)
                if max_free_memory < free_memory:
                    selected_gpu = i
                    max_free_memory = free_memory
            free_memory_mb = max_free_memory / (1024 * 1024)
            if free_memory_mb < self.min_memory:
                logging.warning(
                    f"⚠️ GPU {selected_gpu} has {round(free_memory_mb, 2)} MB memory left. Switching to CPU."
                )
                device = torch.device("cpu")
            else:
                device = torch.device(f"cuda:{selected_gpu}")

        return device

    def custom_logging(default_path="logging.json", default_level=logging.INFO, env_key="LOG_CFG"):
        """
        自定义日志内容
        """
        path = default_path
        value = os.getenv(env_key, None)
        if value:
            path = value
        if os.path.exists(path):
            with open(path, "r") as f:
                config = json.load(f)
                logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)

class export_Model:
    def __init__(self,model=None,model_path="", save_pth=0):
        """
        Args:
            model: Torch.nn
            file_name: export filename
            save_pth: E.g 1? 0  
        """
        self.model = model
        self.model_path = model_path

        self.file_name = model_path.split("\\")[-1]  
        if self.file_name.split('.')[-1] != "pth":
            self.file_name = self.file_name.split('.')[0]+".pth"
        self.model_type = "pth"
        if save_pth==1:torch.save(model, self.file_name)

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

        self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.model.eval()
        self.file_name = self.file_name.split(".")[0]+".onnx"
        self.model_path = self.model_path.split(".")[0]+".onnx"

        torch.onnx.export(
            self.model,
            self.dummy_input,
            self.model_path,
            input_names=[self.input_name],
            output_names=[self.oup_name],
            dynamic_axes={self.input_name: {0: "batch_size"}, self.oup_name: {0: "batch_size"}},
            opset_version=self.opset_version
        )

        self.model_type = "ONNX"

        return self


    def turn_mnn(self,fp16=0, optim=0,bizcode="mobilenet"):
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
        
        Usage:
        .. code-block:: python
            import MNN
            interpreter = MNN.Interpreter(model_path)
            session = interpreter.createSession()
            input_tensor = interpreter.getSessionInput(session)
        """
        if fp16==1:fp="--fp16"
        __command = "mnnconvert -f {model_type} --modelFile {model_path} --MNNModel {model_export_name} --bizCode {bizcode} {fp}  --optimizePrefer {optim}".format(
            model_type=self.model_type,
            model_path=self.file_name,
            model_export_name=self.file_name.split(".")[0]+".mnn",
            bizcode = bizcode,
            fp=fp,
            optim = optim
        )
        import subprocess
        subprocess.run(__command, shell=True, check=True)

        