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


namespace ppyoloe_r_opencvsharp
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
                if (!File.Exists("./model/ppyoloe_r_crn_s_3x_dota.bin") && !File.Exists("./model/ppyoloe_r_crn_s_3x_dota.bin"))
                {
                    if (!File.Exists("./model/ppyoloe_r_crn_s_3x_dota.tar"))
                    {
                        _ = Download.download_file_async("https://github.com/guojin-yan/OpenVINO-CSharp-API-Samples/releases/download/Model/ppyoloe_r_crn_s_3x_dota.tar",
                            "./model/ppyoloe_r_crn_s_3x_dota.tar").Result;
                    }
                    Download.unzip("./model/ppyoloe_r_crn_s_3x_dota.tar", "./model/");
                }

                if (!File.Exists("./model/plane.png"))
                {
                    _ = Download.download_file_async("https://github.com/guojin-yan/OpenVINO-CSharp-API-Samples/releases/download/Image/plane.png",
                        "./model/plane.png").Result;
                }
                model_path = "./model/ppyoloe_r_crn_s_3x_dota.xml";
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

            ppyoloe_r_det(model_path, image_path, device);
        }
        static void ppyoloe_r_det(string model_path, string image_path, string device)
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

            float[] factor = new float[] { 1024.0f / (float)image.Rows, 1024.0f / (float)image.Cols };

            Mat mat = new Mat();
            Cv2.Resize(image, mat, new Size(1024, 1024));
            mat = Normalize.run(mat, new float[] { 0.485f, 0.456f, 0.406f }, new float[] { 1 / 0.229f, 1 / 0.224f, 1 / 0.225f }, true);
            float[] input_data = Permute.run(mat);
            end = DateTime.Now;
            Slog.INFO("5. Process input images success, time spend:" + (end - start).TotalMilliseconds + "ms.");
            // -------- Step 6. Set up input data --------
            start = DateTime.Now;

            Tensor input_tensor_data = infer_request.get_tensor("image");
            input_tensor_data.set_shape(new Shape(1, 3, 1024, 1024));
            input_tensor_data.set_data<float>(input_data);

            Tensor input_tensor_factor = infer_request.get_tensor("scale_factor");
            input_tensor_factor.set_shape(new Shape(1, 2));
            input_tensor_factor.set_data<float>(factor);

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
            Console.WriteLine(output_tensor.get_shape().to_string());
            int output_length = (int)output_tensor.get_size();
            float[] rect_data = output_tensor.get_data<float>(output_length);
            Tensor output_tensor1 = infer_request.get_output_tensor(1);
            Shape conf_shape = output_tensor1.get_shape();
            Console.WriteLine(conf_shape.to_string());
            int output_length1 = (int)output_tensor1.get_size();
            float[] conf_data = output_tensor1.get_data<float>(output_length1);
            end = DateTime.Now;
            Slog.INFO("8. Get infer result data success, time spend:" + (end - start).TotalMilliseconds + "ms.");

            // -------- Step 9. Process reault  --------
            start = DateTime.Now;
            Mat conf_mat = new Mat((int)conf_shape[1], (int)conf_shape[2], MatType.CV_32F, conf_data);
            conf_mat = conf_mat.T();

            
            List<Point2f[]> position_points = new List<Point2f[]>();
            List<RotatedRect> position_boxes = new List<RotatedRect>();
            List<int> class_ids = new List<int>();
            List<float> confidences = new List<float>();

            for (int i = 0; i < conf_mat.Rows; i++)
            {
                Mat classes_scores = new Mat(conf_mat, new Rect(0, i, 15, 1));
                OpenCvSharp.Point max_classId_point, min_classId_point;
                double max_score, min_score;
                // Obtain the maximum value and its position in a set of data
                Cv2.MinMaxLoc(classes_scores, out min_score, out max_score,
                    out min_classId_point, out max_classId_point);
                // Confidence level between 0 ~ 1
                // Obtain identification box information
                if (max_score > 0.25)
                {
                    Point2f[] points = new Point2f[4] { new Point2f(rect_data[8 * i], rect_data[8 * i + 1]),
                        new Point2f(rect_data[8 * i + 2], rect_data[8 * i + 3]),
                        new Point2f(rect_data[8 * i + 4], rect_data[8 * i + 5]),
                        new Point2f(rect_data[8 * i + 6], rect_data[8 * i + 7])};
                    position_boxes.Add(new RotatedRect(new Point2f(rect_data[8 * i], rect_data[8 * i + 1]),
                        new Point2f(rect_data[8 * i + 2], rect_data[8 * i + 3]),
                        new Point2f(rect_data[8 * i + 4], rect_data[8 * i + 5])));
                    position_points.Add(points);
                    class_ids.Add(max_classId_point.X);
                    confidences.Add((float)max_score);
                }
            }

            int[] indexes = new int[position_boxes.Count];
            CvDnn.NMSBoxes(position_boxes, confidences, 0.5f, 0.5f, out indexes);

            end = DateTime.Now;
            Slog.INFO("9. Process reault  success, time spend:" + (end - start).TotalMilliseconds + "ms.");
            for (int n = 0; n < indexes.Length; n++)
            {
                int index = indexes[n];
                Point2f[] points = position_points[index];
                for (int i = 0; i < 4; i++)
                {
                    Cv2.Line(image, (Point)points[i], (Point)points[(i + 1) % 4], new Scalar(255, 100, 200), 2);
                }
     
                Cv2.PutText(image, class_ids[index] + "-" + confidences[index].ToString("0.00"),
                    (Point)points[0],
                    HersheyFonts.HersheySimplex, 0.8, new Scalar(0, 0, 0), 2);
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
