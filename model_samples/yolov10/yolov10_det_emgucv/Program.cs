using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using Emgu.CV;
using OpenVinoSharp.Extensions.model;
using OpenVinoSharp.Extensions.utility;
using OpenVinoSharp.Extensions;
using OpenVinoSharp;
using System.Drawing;
using System.Runtime.InteropServices;

namespace yolov10_det_emgucv
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string model_path = "";
            string image_path = "";
            string device = "AUTO";
            if (args.Length == 0)
            {
                if (!Directory.Exists("./model"))
                {
                    Directory.CreateDirectory("./model");
                }
                if (!File.Exists("./model/yolov10s.bin") && !File.Exists("./model/yolov10s.bin"))
                {
                    if (!File.Exists("./model/yolov10s.tar"))
                    {
                        _ = Download.download_file_async("https://github.com/guojin-yan/OpenVINO-CSharp-API-Samples/releases/download/Model/yolov10s.tar",
                            "./model/yolov10s.tar").Result;
                    }
                    Download.unzip("./model/yolov10s.tar", "./model/");
                }

                if (!File.Exists("./model/test_image.jpg"))
                {
                    _ = Download.download_file_async("https://github.com/guojin-yan/OpenVINO-CSharp-API-Samples/releases/download/Image/test_det_02.jpg",
                        "./model/test_image.jpg").Result;
                }
                model_path = "./model/yolov10s.xml";
                image_path = "./model/test_image.jpg";
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
            Slog.INFO("Start yolov8 model inference.");

            yolov10_det(model_path, image_path, device);
        }

        static void yolov10_det(string model_path, string image_path, string device)
        {
            DateTime start = DateTime.Now;
            // -------- Step 1. Initialize OpenVINO Runtime Core --------
            Core core = new Core();
            DateTime end = DateTime.Now;
            Slog.INFO("1. Initialize OpenVINO Runtime Core success, time spend: " + (end - start).TotalMilliseconds + "ms.");
            // -------- Step 2. Read inference model --------
            start = DateTime.Now;
            OpenVinoSharp.Model model = core.read_model(model_path);
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
            int max_image_length = image.Cols > image.Rows ? image.Cols : image.Rows;
            Mat max_image = Mat.Zeros(max_image_length, max_image_length, DepthType.Cv8U, 3);
            Rectangle roi = new Rectangle(0, 0, image.Cols, image.Rows);
            image.CopyTo(new Mat(max_image, roi));
            float factor = (float)(max_image_length / 640.0);
            end = DateTime.Now;
            Slog.INFO("5. Process input images success, time spend:" + (end - start).TotalMilliseconds + "ms.");
            // -------- Step 6. Set up input data --------
            start = DateTime.Now;
            Tensor input_tensor = infer_request.get_input_tensor();
            Shape input_shape = input_tensor.get_shape();
            Mat input_mat = DnnInvoke.BlobFromImage(max_image, 1.0 / 255.0, new Size((int)input_shape[2], (int)input_shape[3]), new MCvScalar(0), true, false);
            float[] input_data = new float[input_shape[1] * input_shape[2] * input_shape[3]];
            //Marshal.Copy(input_mat.Ptr, input_data, 0, input_data.Length);
            input_mat.CopyTo<float>(input_data);
            input_tensor.set_data<float>(input_data);

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
            Tensor output_tensor = infer_request.get_output_tensor();
            int output_length = (int)output_tensor.get_size();
            float[] output_data = output_tensor.get_data<float>(output_length);
            end = DateTime.Now;
            Slog.INFO("8. Get infer result data success, time spend:" + (end - start).TotalMilliseconds + "ms.");

            // -------- Step 9. Process reault  --------
            start = DateTime.Now;
            // Storage results list
            List<Rectangle> position_boxes = new List<Rectangle>();
            List<int> class_ids = new List<int>();
            List<float> confidences = new List<float>();

            // Preprocessing output results
            for (int i = 0; i < output_data.Length / 6; i++)
            {
                int s = 6 * i;
                if ((float)output_data[s + 4] > 0.5)
                {
                    float cx = output_data[s + 0];
                    float cy = output_data[s + 1];
                    float dx = output_data[s + 2];
                    float dy = output_data[s + 3];
                    int x = (int)((cx) * factor);
                    int y = (int)((cy) * factor);
                    int width = (int)((dx - cx) * factor);
                    int height = (int)((dy - cy) * factor);
                    Rectangle box = new Rectangle();
                    box.X = x;
                    box.Y = y;
                    box.Width = width;
                    box.Height = height;

                    position_boxes.Add(box);
                    class_ids.Add((int)output_data[s + 5]);
                    confidences.Add((float)output_data[s + 4]);
                }
            }
            end = DateTime.Now;
            Slog.INFO("9. Process reault  success, time spend:" + (end - start).TotalMilliseconds + "ms.");
            for (int i = 0; i < class_ids.Count; i++)
            {
                int index = i;
                CvInvoke.Rectangle(image, position_boxes[index], new MCvScalar(0, 0, 255), 2, LineType.Filled);
                CvInvoke.Rectangle(image, new Rectangle(new Point(position_boxes[index].X, position_boxes[index].Y),
                 new Size(position_boxes[index].Width, 30)), new MCvScalar(0, 255, 255), -1);
                CvInvoke.PutText(image, CocoOption.lables[class_ids[index]] + "-" + confidences[index].ToString("0.00"),
                    new Point(position_boxes[index].X, position_boxes[index].Y + 25),
                    FontFace.HersheySimplex, 0.8, new MCvScalar(0, 0, 0), 2);
            }
            string output_path = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(image_path)),
                Path.GetFileNameWithoutExtension(image_path) + "_result.jpg");
            CvInvoke.Imwrite(output_path, image);
            Slog.INFO("The result save to " + output_path);
            CvInvoke.Imshow("Result", image);
            CvInvoke.WaitKey(0);
        }
    }
}
