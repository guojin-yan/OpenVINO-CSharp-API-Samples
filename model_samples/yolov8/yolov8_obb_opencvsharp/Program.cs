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

namespace yolov8_obb_opencvsharp
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
                if (!File.Exists("./model/yolov8s-obb.bin") && !File.Exists("./model/yolov8s-obb.bin"))
                {
                    if (!File.Exists("./model/yolov8s-obb.tar"))
                    {
                        _ = Download.download_file_async("https://github.com/guojin-yan/OpenVINO-CSharp-API-Samples/releases/download/Model/yolov8s-obb.tar",
                            "./model/yolov8s-obb.tar").Result;
                    }
                    Download.unzip("./model/yolov8s-obb.tar", "./model/");
                }

                if (!File.Exists("./model/plane.png"))
                {
                    _ = Download.download_file_async("https://github.com/guojin-yan/OpenVINO-CSharp-API-Samples/releases/download/Image/plane.png",
                        "./model/plane.png").Result;
                }
                model_path = "./model/yolov8s-obb.xml";
                image_path = "./model/plane.png";
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

            yolov8_obb(model_path, image_path, device);
            
        }

        static void yolov8_obb(string model_path, string image_path, string device)
        {
            DateTime start = DateTime.Now;
            // -------- Step 1. Initialize OpenVINO Runtime Core --------
            Core core = new Core();
            DateTime end = DateTime.Now;
            Slog.INFO("1. Initialize OpenVINO Runtime Core success, time spend: " + (end - start).TotalMilliseconds + "ms.");
            // -------- Step 2. Read inference model --------
            start = DateTime.Now;
            Model model = core.read_model(model_path);
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
            Mat max_image = Mat.Zeros(new OpenCvSharp.Size(max_image_length, max_image_length), MatType.CV_8UC3);
            Rect roi = new Rect(0, 0, image.Cols, image.Rows);
            image.CopyTo(new Mat(max_image, roi));
            float factor = (float)(max_image_length / 1024.0);
            end = DateTime.Now;
            Slog.INFO("5. Process input images success, time spend:" + (end - start).TotalMilliseconds + "ms.");
            // -------- Step 6. Set up input data --------
            start = DateTime.Now;
            Tensor input_tensor = infer_request.get_input_tensor();
            Shape input_shape = input_tensor.get_shape();
            Mat input_mat = CvDnn.BlobFromImage(max_image, 1.0 / 255.0, new OpenCvSharp.Size(input_shape[2], input_shape[3]), 0, true, false);
            float[] input_data = new float[input_shape[1] * input_shape[2] * input_shape[3]];
            Marshal.Copy(input_mat.Ptr(0), input_data, 0, input_data.Length);
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
            Mat result_data = new Mat(20, 21504, MatType.CV_32F, output_data);
            result_data = result_data.T();

            float[] d = new float[output_length];
            result_data.GetArray<float>(out d);

            // Storage results list
            List<Rect2d> position_boxes = new List<Rect2d>();
            List<int> class_ids = new List<int>();
            List<float> confidences = new List<float>();
            List<float> rotations = new List<float>();
            // Preprocessing output results
            for (int i = 0; i < result_data.Rows; i++)
            {
                Mat classes_scores = new Mat(result_data, new Rect(4, i, 15, 1));
                OpenCvSharp.Point max_classId_point, min_classId_point;
                double max_score, min_score;
                // Obtain the maximum value and its position in a set of data
                Cv2.MinMaxLoc(classes_scores, out min_score, out max_score,
                    out min_classId_point, out max_classId_point);
                // Confidence level between 0 ~ 1
                // Obtain identification box information
                if (max_score > 0.25)
                {
                    float cx = result_data.At<float>(i, 0);
                    float cy = result_data.At<float>(i, 1);
                    float ow = result_data.At<float>(i, 2);
                    float oh = result_data.At<float>(i, 3);
                    double x = (cx - 0.5 * ow) * factor;
                    double y = (cy - 0.5 * oh) * factor;
                    double width = ow * factor;
                    double height = oh * factor;
                    Rect2d box = new Rect2d();
                    box.X = x;
                    box.Y = y;
                    box.Width = width;
                    box.Height = height;

                    position_boxes.Add(box);
                    class_ids.Add(max_classId_point.X);
                    confidences.Add((float)max_score);
                    rotations.Add(result_data.At<float>(i, 19));
                }
            }
            // NMS non maximum suppression
            int[] indexes = new int[position_boxes.Count];
            CvDnn.NMSBoxes(position_boxes, confidences, 0.25f, 0.7f, out indexes);

            List<RotatedRect> rotated_rects = new List<RotatedRect>();
            for (int i = 0; i < indexes.Length; i++)
            {
                int index = indexes[i];
                
                float w = (float)position_boxes[index].Width;
                float h = (float)position_boxes[index].Height;
                float x = (float)position_boxes[index].X + w / 2;
                float y = (float)position_boxes[index].Y + h / 2;
                float r = rotations[index];
                float w_ = w > h ? w : h;
                float h_ = w > h ? h : w;
                r = (float)((w > h ? r : (float)(r + Math.PI / 2)) % Math.PI);
                RotatedRect rotate = new RotatedRect(new Point2f(x, y), new Size2f(w_, h_), (float)(r * 180.0 / Math.PI));
                rotated_rects.Add(rotate);
            }

            end = DateTime.Now;
            Slog.INFO("9. Process reault  success, time spend:" + (end - start).TotalMilliseconds + "ms.");
            for (int i = 0; i < indexes.Length; i++)
            {
                int index = indexes[i];

                Point2f[] points = rotated_rects[i].Points();
                for (int j = 0; j < 4; j++)
                {
                    Cv2.Line(image, (Point)points[j], (Point)points[(j + 1) % 4], new Scalar(255, 100, 200), 2);
                }
                //Cv2.Rectangle(image, new OpenCvSharp.Point(position_boxes[index].TopLeft.X, position_boxes[index].TopLeft.Y + 30),
                //    new OpenCvSharp.Point(position_boxes[index].BottomRight.X, position_boxes[index].TopLeft.Y), new Scalar(0, 255, 255), -1);
                Cv2.PutText(image, class_lables[class_ids[index]] + "-" + confidences[index].ToString("0.00"),
                    (Point)points[0], HersheyFonts.HersheySimplex, 0.8, new Scalar(0, 0, 0), 2);
            }
            string output_path = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(image_path)),
                Path.GetFileNameWithoutExtension(image_path) + "_result.jpg");
            Cv2.ImWrite(output_path, image);
            Slog.INFO("The result save to " + output_path);
            Cv2.ImShow("Result", image);
            Cv2.WaitKey(0);
        }

        static string[] class_lables = new string[] { "plane","ship", "storage tank", "baseball diamond", "tennis court",
            "basketball court", "ground track field", "harbor",  "bridge",  "large vehicle",  "small vehicle",
            "helicopter",  "roundabout", "soccer ball field", "swimming pool" };
    }
}
