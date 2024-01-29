using OpenVinoSharp.Extensions.utility;
using OpenVinoSharp;
using Emgu.CV;
using Emgu.CV.CvEnum;
using OpenVinoSharp.Extensions;
using System.Runtime.InteropServices;
using System.Drawing;
using System.Diagnostics;
using System.Reflection.PortableExecutable;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using OpenVinoSharp.Extensions.result;
using OpenVinoSharp.Extensions.model;
using OpenVinoSharp.Extensions.process;
using OpenVinoSharp.preprocess;
using System.Reflection;

namespace yolov8_obb_emgucv
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
            //yolov8_det_with_process(model_path, image_path, device);
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
            float factor = (float)(max_image_length / 1024.0);
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
            Mat result_data = new Mat(20, 21504, DepthType.Cv32F, 1,
                           Marshal.UnsafeAddrOfPinnedArrayElement(output_data, 0), 4 * 21504);
            result_data = result_data.T();

            // Storage results list
            List<Rectangle> position_boxes = new List<Rectangle>();
            List<int> class_ids = new List<int>();
            List<float> confidences = new List<float>();
            List<float> rotations = new List<float>();
            // Preprocessing output results
            for (int i = 0; i < result_data.Rows; i++)
            {
                Mat classes_scores = new Mat(result_data, new Rectangle(4, i, 15, 1));//GetArray(i, 5, classes_scores);
                Point max_classId_point = new Point(), min_classId_point = new Point();
                double max_score = 0, min_score = 0;
                // Obtain the maximum value and its position in a set of data
                CvInvoke.MinMaxLoc(classes_scores, ref min_score, ref max_score,
                    ref min_classId_point, ref max_classId_point);
                // Confidence level between 0 ~ 1
                // Obtain identification box information
                if (max_score > 0.25)
                {
                    Mat mat = new Mat(result_data, new Rectangle(0, i, 20, 1));
                    float[,] data = (float[,])mat.GetData();
                    float cx = data[0, 0];
                    float cy = data[0, 1];
                    float ow = data[0, 2];
                    float oh = data[0, 3];
                    int x = (int)((cx - 0.5 * ow) * factor);
                    int y = (int)((cy - 0.5 * oh) * factor);
                    int width = (int)(ow * factor);
                    int height = (int)(oh * factor);
                    Rectangle box = new Rectangle();
                    box.X = x;
                    box.Y = y;
                    box.Width = width;
                    box.Height = height;

                    position_boxes.Add(box);
                    class_ids.Add(max_classId_point.X);
                    confidences.Add((float)max_score);
                    rotations.Add(data[0, 19]);
                }
            }

            // NMS non maximum suppression
            int[] indexes = DnnInvoke.NMSBoxes(position_boxes.ToArray(), confidences.ToArray(), 0.5f, 0.5f);

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
                RotatedRect rotate = new RotatedRect(new PointF(x, y), new SizeF(w_, h_), (float)(r * 180.0 / Math.PI));
                rotated_rects.Add(rotate);
            }


            end = DateTime.Now;
            Slog.INFO("9. Process reault  success, time spend:" + (end - start).TotalMilliseconds + "ms.");
            for (int i = 0; i < indexes.Length; i++)
            {
                int index = indexes[i];

                PointF[] points = rotated_rects[i].GetVertices();
                for (int j = 0; j < 4; j++)
                {
                    CvInvoke.Line(image, new Point((int)points[j].X, (int)points[j].Y), 
                        new Point((int)points[(j + 1) % 4].X, (int)points[(j + 1) % 4].Y), new MCvScalar(255, 100, 200), 2);
                }
                //Cv2.Rectangle(image, new OpenCvSharp.Point(position_boxes[index].TopLeft.X, position_boxes[index].TopLeft.Y + 30),
                //    new OpenCvSharp.Point(position_boxes[index].BottomRight.X, position_boxes[index].TopLeft.Y), new Scalar(0, 255, 255), -1);
                CvInvoke.PutText(image, class_lables[class_ids[index]] + "-" + confidences[index].ToString("0.00"),
                    new Point((int)points[0].X, (int)points[0].Y), FontFace.HersheySimplex, 0.8, new MCvScalar(0, 0, 0), 2);
            }
            string output_path = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(image_path)),
                Path.GetFileNameWithoutExtension(image_path) + "_result.jpg");
            CvInvoke.Imwrite(output_path, image);
            Slog.INFO("The result save to " + output_path);
            CvInvoke.Imshow("Result", image);
            CvInvoke.WaitKey(0);
        }
        static string[] class_lables = new string[] { "plane","ship", "storage tank", "baseball diamond", "tennis court",
            "basketball court", "ground track field", "harbor",  "bridge",  "large vehicle",  "small vehicle",
            "helicopter",  "roundabout", "soccer ball field", "swimming pool" };
    }
}
