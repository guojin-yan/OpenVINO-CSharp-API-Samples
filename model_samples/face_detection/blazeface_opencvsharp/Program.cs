using OpenCvSharp.Dnn;
using OpenCvSharp;
using OpenVinoSharp;
using OpenVinoSharp.Extensions;
using OpenVinoSharp.Extensions.utility;
using System.Runtime.InteropServices;
using OpenVinoSharp.preprocess;
using OpenVinoSharp.Extensions.result;
using OpenVinoSharp.Extensions.process;
using System;
using System.Reflection.Metadata;

namespace blazeface_opencvsharp
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string model_path = "";
            string image_path = "";
            string device = "CPU";
            if (args.Length == 0)
            {
                if (!Directory.Exists("./model"))
                {
                    Directory.CreateDirectory("./model");
                }
                if (!File.Exists("./model/blazeface_1000e.xml")
                    && !File.Exists("./model/blazeface_1000e.bin"))
                {
                    if (!File.Exists("./model/blazeface_1000e.tar"))
                    {
                        _ = Download.download_file_async("https://github.com/guojin-yan/OpenVINO-CSharp-API-Samples/releases/download/Model/blazeface_1000e.tar",
                            "./model/blazeface_1000e.tar").Result;
                    }
                    Download.unzip("./model/blazeface_1000e.tar", "./model/");
                }

                if (!File.Exists("./model/face1.jpg"))
                {
                    _ = Download.download_file_async("https://github.com/guojin-yan/OpenVINO-CSharp-API-Samples/releases/download/Image/face1.jpg",
                        "./model/face1.jpg").Result;
                }
                model_path = "./model/blazeface_1000e.xml";
                image_path = "./model/face1.jpg";
            }
            else if (args.Length >= 2)
            {
                model_path = args[0];
                image_path = args[1];
                device = args[2];
            }
            else
            {
                Console.WriteLine("Please enter the correct command parameters, for example:");
                Console.WriteLine("> 1. dotnet run");
                Console.WriteLine("> 2. dotnet run <model path> <image path> <device name>");
            }
            // -------- Get OpenVINO runtime version --------

            OpenVinoSharp.Version version = Ov.get_openvino_version();

            Slog.INFO("---- OpenVINO INFO----");
            Slog.INFO("Description : " + version.description);
            Slog.INFO("Build number: " + version.buildNumber);

            Slog.INFO("Predict model files: " + model_path);
            Slog.INFO("Predict image  files: " + image_path);
            Slog.INFO("Inference device: " + device);
            Slog.INFO("Start RT-DETR model inference.");

            face_detection(model_path, image_path, device);

        }
        static void face_detection(string model_path, string image_path, string device)
        {
            DateTime start = DateTime.Now;
            // -------- Step 1. Initialize OpenVINO Runtime Core --------
            Core core = new Core();
            DateTime end = DateTime.Now;
            Slog.INFO("1. Initialize OpenVINO Runtime Core success, time spend: " + (end - start).TotalMilliseconds + "ms.");
            // -------- Step 2. Read inference model --------
            start = DateTime.Now;
            Model model = core.read_model(model_path);

            Dictionary<string,PartialShape> pairs = new Dictionary<string,PartialShape>();
            pairs.Add("scale_factor", new PartialShape(new Shape(1, 2)));
            pairs.Add("im_shape", new PartialShape(new Shape(1, 2)));
            pairs.Add("image", new PartialShape(new Shape(1, 3, 640, 640)));

            model.reshape(pairs);

            end = DateTime.Now;
            Slog.INFO("2. Read inference model success, time spend: " + (end - start).TotalMilliseconds + "ms.");
            OvExtensions.printf_model_info(model);
            // -------- Step 3. Loading a model to the device --------
            start = DateTime.Now;
            CompiledModel compiled_model = core.compile_model(model, device);
            end = DateTime.Now;
            Slog.INFO("3. Loading a model to the device success, time spend:" + (end - start).TotalMilliseconds + "ms.");
            // -------- Step 4. Create an infer request --------
            start = DateTime.Now;
            InferRequest infer_request = compiled_model.create_infer_request();
            end = DateTime.Now;
            Slog.INFO("4. Create an infer request success, time spend:" + (end - start).TotalMilliseconds + "ms.");
            // -------- Step 5. Process input images --------
            start = DateTime.Now;
            Mat image = new Mat(image_path); // Read image by opencvsharp
            //Cv2.ImShow("ss", image);
            //Cv2.WaitKey(0);
            Mat mat = new Mat();
            Cv2.Resize(image, mat, new Size(640, 640));
            mat = Normalize.run(mat, new float[] { 123f, 117f, 104f }, new float[] {1/127.502231f, 1/127.502231f, 1/127.502231f }, 
                false);
            float[] input_data = Permute.run(mat);
            end = DateTime.Now;
            Slog.INFO("5. Process input images success, time spend:" + (end - start).TotalMilliseconds + "ms.");
            // -------- Step 6. Set up input data --------
            start = DateTime.Now;

            Tensor input_tensor_data = infer_request.get_tensor("image");
            //input_tensor_data.set_shape(new Shape(1, 3, image.Cols, image.Rows));
            input_tensor_data.set_data<float>(input_data);
            Tensor input_tensor_shape = infer_request.get_tensor("im_shape");
            input_tensor_shape.set_shape(new Shape(1, 2));
            input_tensor_shape.set_data<float>(new float[] { 640,640 });
            Tensor input_tensor_factor = infer_request.get_tensor("scale_factor");
            input_tensor_factor.set_shape(new Shape(1, 2));
            input_tensor_factor.set_data<float>(new float[] { ((float)640.0f / image.Rows),((float)640.0/image.Cols) });

            end = DateTime.Now;
            Slog.INFO("6. Set up input data success, time spend:" + (end - start).TotalMilliseconds + "ms.");
            // -------- Step 7. Do inference synchronously --------
            infer_request.infer();
            start = DateTime.Now;
            infer_request.infer();
            end = DateTime.Now;
            Slog.INFO("7. Do inference synchronously success, time spend:" + (end - start).TotalMilliseconds + "ms.");
            // -------- Step 8. Get infer result data --------
            start = DateTime.Now;
            Tensor output_tensor = infer_request.get_output_tensor(0);
            Shape output_shape = output_tensor.get_shape();
            int output_length = (int)output_tensor.get_size();
            float[] result_data = output_tensor.get_data<float>(output_length);
            Tensor output_tensor1 = infer_request.get_output_tensor(1);
            int output_length1 = (int)output_tensor1.get_size();
            int[] result_len = output_tensor1.get_data<int>(output_length1);
            end = DateTime.Now;
            Slog.INFO("8. Get infer result data success, time spend:" + (end - start).TotalMilliseconds + "ms.");

            // -------- Step 9. Process reault  --------
            start = DateTime.Now;
            List<Rect> position_boxes = new List<Rect>();
            List<float> confidences = new List<float>();
            // Preprocessing output results
            for (int i = 0; i < result_len[0]; i++)
            {
                double confidence = result_data[6 * i + 1];
                if (confidence > 0.5)
                {
                    float tlx = result_data[6 * i + 2];
                    float tly = result_data[6 * i + 3];
                    float brx = result_data[6 * i + 4];
                    float bry = result_data[6 * i + 5];
                    Rect box = new Rect((int)tlx, (int)tly, (int)(brx - tlx), (int)(bry - tly));
                    position_boxes.Add(box);
                    confidences.Add((float)confidence);
                }
            }

            end = DateTime.Now;
            Slog.INFO("9. Process reault  success, time spend:" + (end - start).TotalMilliseconds + "ms.");
            for (int i = 0; i < position_boxes.Count; i++)
            {
                int index = i;
                Cv2.Rectangle(image, position_boxes[index], new Scalar(255, 0, 0), 1, LineTypes.Link8);
                Cv2.PutText(image, confidences[index].ToString("0.00"),
                    new OpenCvSharp.Point(position_boxes[index].TopLeft.X, position_boxes[index].TopLeft.Y-5),
                    HersheyFonts.HersheySimplex, 0.4, new Scalar(255, 0, 0), 1);
            }
            string output_path = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(image_path)),
                Path.GetFileNameWithoutExtension(image_path) + "_result.jpg");
            Cv2.ImWrite(output_path, image);
            Slog.INFO("The result save to " + output_path);
            Cv2.ImShow("Result", image);
            Cv2.WaitKey(0);
        }
    }
}
