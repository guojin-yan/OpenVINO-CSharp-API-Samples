using OpenVinoSharp.Extensions.utility;
using OpenVinoSharp;
using Emgu.CV.CvEnum;
using Emgu.CV;
using OpenVinoSharp.Extensions;
using System.Runtime.InteropServices;
using Emgu.CV.Structure;
using OpenVinoSharp.Extensions.model;
using System.Drawing;
using Emgu.CV.Dnn;
using OpenVinoSharp.Extensions.process;
using OpenVinoSharp.Extensions.result;
using OpenVinoSharp.preprocess;

namespace rt_detr_emgucv
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
                //if (!File.Exists("./model/rtdetr_r50vd_6x_coco.xml") 
                //    && !File.Exists("./model/rtdetr_r50vd_6x_coco.bin"))
                //{
                //    if (!File.Exists("./model/rtdetr_r50vd_6x_coco.tar"))
                //    {
                //        _ = Download.download_file_async("https://bj.bcebos.com/v1/paddle-slim-models/act/rtdetr_r50vd_6x_coco_quant.tar",
                //            "./model/rtdetr_r50vd_6x_coco.tar").Result;
                //    }
                //    Download.unzip("./model/rtdetr_r50vd_6x_coco.tar", "./model/");
                //}

                if (!File.Exists("./model/test_image.jpg"))
                {
                    _ = Download.download_file_async("https://github.com/guojin-yan/OpenVINO-CSharp-API-Samples/releases/download/Image/test_det_02.jpg",
                        "./model/test_image.jpg").Result;
                }
                //model_path = "./model/rtdetr_r50vd_6x_coco_quant/model.pdmodel";
                model_path = "./model/rtdetr_r50vd_6x_coco.xml";

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
            Slog.INFO("Start RT-DETR model inference.");

            //rtdetr_det(model_path, image_path, device);
            //rtdetr_det_with_process(model_path, image_path, device);
            RTDETR_det_using_extensions(model_path, image_path, device);
        }

        static void rtdetr_det(string model_path, string image_path, string device)
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

            float[] factor = new float[] { 640.0f / (float)image.Rows, 640.0f / (float)image.Cols };
            float[] im_shape = new float[] { 640.0f, 640.0f };

            Mat input_mat = DnnInvoke.BlobFromImage(image, 1.0 / 255.0, new Size(640, 640), new MCvScalar(0), true, false);
            float[] input_data = new float[640 * 640 * 3];
            input_mat.CopyTo<float>(input_data);
            end = DateTime.Now;
            Slog.INFO("5. Process input images success, time spend:" + (end - start).TotalMilliseconds + "ms.");
            // -------- Step 6. Set up input data --------
            start = DateTime.Now;
            Tensor input_tensor_shape = infer_request.get_tensor("im_shape");
            input_tensor_shape.set_shape(new Shape(1, 2));
            input_tensor_shape.set_data(im_shape);

            Tensor input_tensor_data = infer_request.get_tensor("image");
            input_tensor_data.set_shape(new Shape(1, 3, 640, 640));
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
            int output_length = (int)output_tensor.get_size();
            float[] output_data = output_tensor.get_data<float>(output_length);
            end = DateTime.Now;
            Slog.INFO("8. Get infer result data success, time spend:" + (end - start).TotalMilliseconds + "ms.");

            // -------- Step 9. Process reault  --------
            start = DateTime.Now;
            List<Rectangle> position_boxes = new List<Rectangle>();
            List<int> class_ids = new List<int>();
            List<float> confidences = new List<float>();

            for (int i = 0; i < 300; ++i)
            {
                if (output_data[6 * i + 1] > 0.5)
                {
                    class_ids.Add((int)output_data[6 * i]);
                    confidences.Add(output_data[6 * i + 1]);
                    position_boxes.Add(new Rectangle((int)output_data[6 * i + 2], (int)output_data[6 * i + 3],
                        (int)(output_data[6 * i + 4] - output_data[6 * i + 2]),
                        (int)(output_data[6 * i + 5] - output_data[6 * i + 3])));
                }
            }
            end = DateTime.Now;
            Slog.INFO("9. Process reault  success, time spend:" + (end - start).TotalMilliseconds + "ms.");
            for (int index = 0; index < class_ids.Count; index++)
            {
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
        static void rtdetr_det_with_process(string model_path, string image_path, string device)
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
            InputInfo input_info = processor.input("image");
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
            Mat input_image = new Mat();
            CvInvoke.Resize(image, input_image, new Size(640, 640));
            float[] factor = new float[] { 640.0f / (float)image.Rows, 640.0f / (float)image.Cols };
            float[] im_shape = new float[] { 640.0f, 640.0f };
            end = DateTime.Now;
            Slog.INFO("5. Process input images success, time spend:" + (end - start).TotalMilliseconds + "ms.");
            // -------- Step 6. Set up input data --------
            start = DateTime.Now;
            Tensor input_tensor_shape = infer_request.get_tensor("im_shape");
            input_tensor_shape.set_shape(new Shape(1, 2));
            input_tensor_shape.set_data(im_shape);

            Tensor input_tensor_data = infer_request.get_tensor("image");

            byte[] input_data = new byte[3 * 640 * 640];
            //max_image.GetArray<int>(out input_data);
            input_image.CopyTo<byte>(input_data);
            IntPtr destination = input_tensor_data.data();
            Marshal.Copy(input_data, 0, destination, input_data.Length);
            //input_tensor.set_data<byte>(input_data);

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
            int output_length = (int)output_tensor.get_size();
            float[] output_data = output_tensor.get_data<float>(output_length);
            end = DateTime.Now;
            Slog.INFO("8. Get infer result data success, time spend:" + (end - start).TotalMilliseconds + "ms.");

            // -------- Step 9. Process reault  --------
            start = DateTime.Now;
            List<Rectangle> position_boxes = new List<Rectangle>();
            List<int> class_ids = new List<int>();
            List<float> confidences = new List<float>();

            for (int i = 0; i < 300; ++i)
            {
                if (output_data[6 * i + 1] > 0.5)
                {
                    class_ids.Add((int)output_data[6 * i]);
                    confidences.Add(output_data[6 * i + 1]);
                    position_boxes.Add(new Rectangle((int)output_data[6 * i + 2], (int)output_data[6 * i + 3],
                        (int)(output_data[6 * i + 4] - output_data[6 * i + 2]),
                        (int)(output_data[6 * i + 5] - output_data[6 * i + 3])));
                }
            }
            end = DateTime.Now;
            Slog.INFO("9. Process reault  success, time spend:" + (end - start).TotalMilliseconds + "ms.");
            for (int index = 0; index < class_ids.Count; index++)
            {
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

        static void RTDETR_det_using_extensions(string model_path, string image_path, string device)
        {
            RtdetrConfig config = new RtdetrConfig();
            config.set_model(model_path);
            RtdetrDet yolov8 = new RtdetrDet(config);
            Mat image = CvInvoke.Imread(image_path);
            DetResult result = yolov8.predict(image);
            Mat result_im = Visualize.draw_det_result(result, image);
            CvInvoke.Imshow("Result", result_im);
            CvInvoke.WaitKey(0);
        }

    }
}
