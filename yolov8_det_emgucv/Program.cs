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

namespace yolov8_det_emgucv
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
                if (!File.Exists("./model/yolov8s.bin") && !File.Exists("./model/yolov8s.bin"))
                {
                    if (!File.Exists("./model/yolov8s.tar"))
                    {
                        _ = Download.download_file_async("https://github.com/guojin-yan/OpenVINO-CSharp-API-Samples/releases/download/Model/yolov8s.tar",
                            "./model/yolov8s.tar").Result;
                    }
                    Download.unzip("./model/yolov8s.tar", "./model/");
                }

                if (!File.Exists("./model/test_image.jpg"))
                {
                    _ = Download.download_file_async("https://github.com/guojin-yan/OpenVINO-CSharp-API-Samples/releases/download/Image/test_det_02.jpg",
                        "./model/test_image.jpg").Result;
                }
                model_path = "./model/yolov8s.xml";
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

            //yolov8_det(model_path, image_path, device);
            //yolov8_det_with_process(model_path, image_path, device);
            yolov8_det_using_extensions(model_path, image_path, device);
        }

        static void yolov8_det(string model_path, string image_path, string device)
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
            Mat result_data = new Mat(84, 8400, DepthType.Cv32F, 1,
                           Marshal.UnsafeAddrOfPinnedArrayElement(output_data, 0), 4 * 8400);
            result_data = result_data.T();

            // Storage results list
            List<Rectangle> position_boxes = new List<Rectangle>();
            List<int> class_ids = new List<int>();
            List<float> confidences = new List<float>();
            // Preprocessing output results
            for (int i = 0; i < result_data.Rows; i++)
            {
                Mat classes_scores = new Mat(result_data, new Rectangle(4, i, 80, 1));//GetArray(i, 5, classes_scores);
                Point max_classId_point = new Point(), min_classId_point = new Point();
                double max_score = 0, min_score = 0;
                // Obtain the maximum value and its position in a set of data
                CvInvoke.MinMaxLoc(classes_scores, ref min_score, ref max_score,
                    ref min_classId_point, ref max_classId_point);
                // Confidence level between 0 ~ 1
                // Obtain identification box information
                if (max_score > 0.25)
                {
                    Mat mat = new Mat(result_data, new Rectangle(0, i, 4, 1));
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
                }
            }

            // NMS non maximum suppression
            int[] indexes = DnnInvoke.NMSBoxes(position_boxes.ToArray(), confidences.ToArray(), 0.5f, 0.5f);
            end = DateTime.Now;
            Slog.INFO("9. Process reault  success, time spend:" + (end - start).TotalMilliseconds + "ms.");
            for (int i = 0; i < indexes.Length; i++)
            {
                int index = indexes[i];
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

        static void yolov8_det_with_process(string model_path, string image_path, string device)
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

            PrePostProcessor processor = new PrePostProcessor(model);

            Tensor input_tensor_pro = new Tensor(new OvType(ElementType.U8), new Shape(1, 640, 640, 3));
            InputInfo input_info = processor.input(0);
            InputTensorInfo input_tensor_info = input_info.tensor();
            input_tensor_info.set_from(input_tensor_pro).set_layout(new Layout("NHWC")).set_color_format(ColorFormat.BGR);

            PreProcessSteps process_steps = input_info.preprocess();
            process_steps.convert_color(ColorFormat.RGB).resize(ResizeAlgorithm.RESIZE_LINEAR)
                .convert_element_type(new OvType(ElementType.F32)).scale(255.0f).convert_layout(new Layout("NCHW"));

            OpenVinoSharp.Model new_model = processor.build();
            // -------- Step 3. Loading a model to the device --------
            start = DateTime.Now;
            CompiledModel compiled_model = core.compile_model(new_model, device);
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
            CvInvoke.Resize(max_image, max_image, new Size(640, 640));
            float factor = (float)(max_image_length / 640.0);
            end = DateTime.Now;
            Slog.INFO("5. Process input images success, time spend:" + (end - start).TotalMilliseconds + "ms.");
            // -------- Step 6. Set up input data --------
            start = DateTime.Now;
            Tensor input_tensor = infer_request.get_input_tensor();
            Shape input_shape = input_tensor.get_shape();
            byte[] input_data = new byte[input_shape[1] * input_shape[2] * input_shape[3]];
            //max_image.GetArray<int>(out input_data);
            max_image.CopyTo<byte>(input_data);
            IntPtr destination = input_tensor.data();
            Marshal.Copy(input_data, 0, destination, input_data.Length);
            //input_tensor.set_data<byte>(input_data);

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
            Mat result_data = new Mat(84, 8400, DepthType.Cv32F, 1,
                           Marshal.UnsafeAddrOfPinnedArrayElement(output_data, 0), 4 * 8400);
            result_data = result_data.T();

            // Storage results list
            List<Rectangle> position_boxes = new List<Rectangle>();
            List<int> class_ids = new List<int>();
            List<float> confidences = new List<float>();
            // Preprocessing output results
            for (int i = 0; i < result_data.Rows; i++)
            {
                Mat classes_scores = new Mat(result_data, new Rectangle(4, i, 80, 1));//GetArray(i, 5, classes_scores);
                Point max_classId_point = new Point(), min_classId_point = new Point();
                double max_score = 0, min_score = 0;
                // Obtain the maximum value and its position in a set of data
                CvInvoke.MinMaxLoc(classes_scores, ref min_score, ref max_score,
                    ref min_classId_point, ref max_classId_point);
                // Confidence level between 0 ~ 1
                // Obtain identification box information
                if (max_score > 0.25)
                {
                    Mat mat = new Mat(result_data, new Rectangle(0, i, 4, 1));
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
                }
            }

            // NMS non maximum suppression
            int[] indexes = DnnInvoke.NMSBoxes(position_boxes.ToArray(), confidences.ToArray(), 0.5f, 0.5f);
            end = DateTime.Now;
            Slog.INFO("9. Process reault  success, time spend:" + (end - start).TotalMilliseconds + "ms.");
            for (int i = 0; i < indexes.Length; i++)
            {
                int index = indexes[i];
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


        static void yolov8_det_using_extensions(string model_path, string image_path, string device)
        {
            Yolov8DetConfig config = new Yolov8DetConfig();
            config.set_model(model_path);
            Yolov8Det yolov8 = new Yolov8Det(config);
            Mat image = CvInvoke.Imread(image_path);
            DetResult result = yolov8.predict(image);
            Mat result_im = Visualize.draw_det_result(result, image);
            CvInvoke.Imshow("Result", result_im);
            CvInvoke.WaitKey(0);
        }

    }
}
