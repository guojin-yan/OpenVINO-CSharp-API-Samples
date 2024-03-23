using OpenCvSharp.Dnn;
using OpenCvSharp;
using OpenVinoSharp.Extensions.utility;
using OpenVinoSharp.Extensions;
using OpenVinoSharp.preprocess;
using OpenVinoSharp;
using System.Runtime.InteropServices;
using OpenVinoSharp.Extensions.model;
using OpenVinoSharp.Extensions.result;

namespace yolov9_seg_opencvsharp
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
                if (!File.Exists("./model/yolov9-c-seg.xml") && !File.Exists("./model/yolov9-c-seg.bin"))
                {
                    if (!File.Exists("./model/yolov9-c-seg.tar"))
                    {
                        _ = Download.download_file_async("https://github.com/guojin-yan/OpenVINO-CSharp-API-Samples/releases/download/Model/yolov9-c-seg.tar",
                            "./model/yolov9-c-seg.tar").Result;
                    }
                    Download.unzip("./model/yolov9-c-seg.tar", "./model/");
                }

                if (!File.Exists("./model/test_det_01.jpg"))
                {
                    _ = Download.download_file_async("https://github.com/guojin-yan/OpenVINO-CSharp-API-Samples/releases/download/Image/test_det_03.jpg",
                        "./model/test_det_01.jpg").Result;
                }
                model_path = "./model/yolov9-c-seg.xml";
                image_path = "./model/test_det_01.jpg";
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
            Slog.INFO("Start yolov9 model inference.");


            //yolov9_seg(model_path, image_path, device);
            yolov9_seg_with_process(model_path, image_path, device);
        }
        static void yolov9_seg(string model_path, string image_path, string device)
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
            float factor = (float)(max_image_length / 640.0);
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

            Tensor output_tensor_0 = infer_request.get_output_tensor(0);
            float[] result_detect = output_tensor_0.get_data<float>((int)output_tensor_0.get_size());

            Tensor output_tensor_1 = infer_request.get_output_tensor(1);
            float[] result_proto = output_tensor_1.get_data<float>((int)output_tensor_1.get_size());

            Mat detect_data = new Mat(116 , 8400, MatType.CV_32FC1, result_detect);
            Mat proto_data = new Mat(32, 25600, MatType.CV_32F, result_proto);
            detect_data = detect_data.T();
            List<Rect> position_boxes = new List<Rect>();
            List<int> class_ids = new List<int>();
            List<float> confidences = new List<float>();
            List<Mat> masks = new List<Mat>();
            for (int i = 0; i < detect_data.Rows; i++)
            {

                Mat classes_scores = new Mat(detect_data, new Rect(4, i, 80, 1));//GetArray(i, 5, classes_scores);
                Point max_classId_point, min_classId_point;
                double max_score, min_score;
                Cv2.MinMaxLoc(classes_scores, out min_score, out max_score,
                    out min_classId_point, out max_classId_point);

                if (max_score > 0.25)
                {
                    //Console.WriteLine(max_score);

                    Mat mask = new Mat(detect_data, new Rect(4 + 80, i, 32, 1));//detect_data.Row(i).ColRange(4 + categ_nums, categ_nums + 36);

                    float cx = detect_data.At<float>(i, 0);
                    float cy = detect_data.At<float>(i, 1);
                    float ow = detect_data.At<float>(i, 2);
                    float oh = detect_data.At<float>(i, 3);
                    int x = (int)((cx - 0.5 * ow) * factor);
                    int y = (int)((cy - 0.5 * oh) * factor);
                    int width = (int)(ow * factor);
                    int height = (int)(oh * factor);
                    Rect box = new Rect();
                    box.X = x;
                    box.Y = y;
                    box.Width = width;
                    box.Height = height;

                    position_boxes.Add(box);
                    class_ids.Add(max_classId_point.X);
                    confidences.Add((float)max_score);
                    masks.Add(mask);
                }
            }


            int[] indexes = new int[position_boxes.Count];
            CvDnn.NMSBoxes(position_boxes, confidences, 0.5f, 0.5f, out indexes);

            SegResult result = new SegResult();
            Mat rgb_mask = Mat.Zeros(new Size((int)image.Size().Width, (int)image.Size().Height), MatType.CV_8UC3);
            Random rd = new Random(); // Generate Random Numbers
            for (int i = 0; i < indexes.Length; i++)
            {
                int index = indexes[i];
                // Division scope
                Rect box = position_boxes[index];
                int box_x1 = Math.Max(0, box.X);
                int box_y1 = Math.Max(0, box.Y);
                int box_x2 = Math.Max(0, box.BottomRight.X);
                int box_y2 = Math.Max(0, box.BottomRight.Y);

                // Segmentation results
                Mat original_mask = masks[index] * proto_data;
                for (int col = 0; col < original_mask.Cols; col++)
                {
                    original_mask.Set<float>(0, col, sigmoid(original_mask.At<float>(0, col)));
                }
                // 1x25600 -> 160x160 Convert to original size
                Mat reshape_mask = original_mask.Reshape(1, 160);

                //Console.WriteLine("m1.size = {0}", m1.Size());

                // Split size after scaling
                int mx1 = Math.Max(0, (int)((box_x1 / factor) * 0.25));
                int mx2 = Math.Min(160, (int)((box_x2 / factor) * 0.25));
                int my1 = Math.Max(0, (int)((box_y1 / factor) * 0.25));
                int my2 = Math.Min(160, (int)((box_y2 / factor) * 0.25));
                // Crop Split Region
                Mat mask_roi = new Mat(reshape_mask, new OpenCvSharp.Range(my1, my2), new OpenCvSharp.Range(mx1, mx2));
                // Convert the segmented area to the actual size of the image
                Mat actual_maskm = new Mat();
                Cv2.Resize(mask_roi, actual_maskm, new Size(box_x2 - box_x1, box_y2 - box_y1));
                // Binary segmentation region
                for (int r = 0; r < actual_maskm.Rows; r++)
                {
                    for (int c = 0; c < actual_maskm.Cols; c++)
                    {
                        float pv = actual_maskm.At<float>(r, c);
                        if (pv > 0.5)
                        {
                            actual_maskm.Set<float>(r, c, 1.0f);
                        }
                        else
                        {
                            actual_maskm.Set<float>(r, c, 0.0f);
                        }
                    }
                }

                // 预测
                Mat bin_mask = new Mat();
                actual_maskm = actual_maskm * 200;
                actual_maskm.ConvertTo(bin_mask, MatType.CV_8UC1);
                if ((box_y1 + bin_mask.Rows) >= (int)image.Size().Height)
                {
                    box_y2 = (int)image.Size().Height - 1;
                }
                if ((box_x1 + bin_mask.Cols) >= (int)image.Size().Width)
                {
                    box_x2 = (int)image.Size().Width - 1;
                }
                // Obtain segmentation area
                Mat mask = Mat.Zeros(new Size((int)image.Size().Width, (int)image.Size().Height), MatType.CV_8UC1);
                bin_mask = new Mat(bin_mask, new OpenCvSharp.Range(0, box_y2 - box_y1), new OpenCvSharp.Range(0, box_x2 - box_x1));
                Rect roi1 = new Rect(box_x1, box_y1, box_x2 - box_x1, box_y2 - box_y1);
                bin_mask.CopyTo(new Mat(mask, roi1));
                // Color segmentation area
                Cv2.Add(rgb_mask, new Scalar(rd.Next(0, 255), rd.Next(0, 255), rd.Next(0, 255)), rgb_mask, mask);
                result.add(class_ids[index], confidences[index], position_boxes[index], rgb_mask.Clone());
            }

            Mat masked_img = new Mat();
            // Draw recognition results on the image
            for (int i = 0; i < result.count; i++)
            {
                Cv2.Rectangle(image, result.datas[i].box, new Scalar(0, 0, 255), 2, LineTypes.Link8);
                Cv2.Rectangle(image, new Point(result.datas[i].box.TopLeft.X, result.datas[i].box.TopLeft.Y + 30),
                    new Point(result.datas[i].box.BottomRight.X, result.datas[i].box.TopLeft.Y), new Scalar(0, 255, 255), -1);
                Cv2.PutText(image, CocoOption.lables[result.datas[i].index] + "-" + result.datas[i].score.ToString("0.00"),
                    new Point(result.datas[i].box.X, result.datas[i].box.Y + 25),
                    HersheyFonts.HersheySimplex, 0.8, new Scalar(0, 0, 0), 2);
                Cv2.AddWeighted(image, 0.5, result.datas[i].mask, 0.5, 0, masked_img);
            }


            string output_path = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(image_path)),
                Path.GetFileNameWithoutExtension(image_path) + "_result.jpg");
            Cv2.ImWrite(output_path, masked_img);
            Slog.INFO("The result save to " + output_path);
            Cv2.ImShow("Result", masked_img);
            Cv2.WaitKey(0);

        }

        static void yolov9_seg_with_process(string model_path, string image_path, string device)
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

            PrePostProcessor processor = new PrePostProcessor(model);

            Tensor input_tensor_pro = new Tensor(new OvType(ElementType.U8), new Shape(1, 640, 640, 3));
            InputInfo input_info = processor.input(0);
            InputTensorInfo input_tensor_info = input_info.tensor();
            input_tensor_info.set_from(input_tensor_pro).set_layout(new Layout("NHWC")).set_color_format(ColorFormat.BGR);

            PreProcessSteps process_steps = input_info.preprocess();
            process_steps.convert_color(ColorFormat.RGB).resize(ResizeAlgorithm.RESIZE_LINEAR)
                .convert_element_type(new OvType(ElementType.F32)).scale(255.0f).convert_layout(new Layout("NCHW"));

            Model new_model = processor.build();
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
            Mat max_image = Mat.Zeros(new OpenCvSharp.Size(max_image_length, max_image_length), MatType.CV_8UC3);
            Rect roi = new Rect(0, 0, image.Cols, image.Rows);
            image.CopyTo(new Mat(max_image, roi));
            Cv2.Resize(max_image, max_image, new OpenCvSharp.Size(640, 640));
            float factor = (float)(max_image_length / 640.0);
            end = DateTime.Now;
            Slog.INFO("5. Process input images success, time spend:" + (end - start).TotalMilliseconds + "ms.");
            // -------- Step 6. Set up input data --------
            start = DateTime.Now;
            Tensor input_tensor = infer_request.get_input_tensor();
            Shape input_shape = input_tensor.get_shape();
            byte[] input_data = new byte[input_shape[1] * input_shape[2] * input_shape[3]];
            //max_image.GetArray<int>(out input_data);
            Marshal.Copy(max_image.Ptr(0), input_data, 0, input_data.Length);
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
            Tensor output_tensor_0 = infer_request.get_output_tensor(0);
            float[] result_detect = output_tensor_0.get_data<float>((int)output_tensor_0.get_size());

            Tensor output_tensor_1 = infer_request.get_output_tensor(1);
            float[] result_proto = output_tensor_1.get_data<float>((int)output_tensor_1.get_size());

            Mat detect_data = new Mat(116, 8400, MatType.CV_32FC1, result_detect);
            Mat proto_data = new Mat(32, 25600, MatType.CV_32F, result_proto);
            detect_data = detect_data.T();
            List<Rect> position_boxes = new List<Rect>();
            List<int> class_ids = new List<int>();
            List<float> confidences = new List<float>();
            List<Mat> masks = new List<Mat>();
            for (int i = 0; i < detect_data.Rows; i++)
            {

                Mat classes_scores = new Mat(detect_data, new Rect(4, i, 80, 1));//GetArray(i, 5, classes_scores);
                Point max_classId_point, min_classId_point;
                double max_score, min_score;
                Cv2.MinMaxLoc(classes_scores, out min_score, out max_score,
                    out min_classId_point, out max_classId_point);

                if (max_score > 0.25)
                {
                    //Console.WriteLine(max_score);

                    Mat mask = new Mat(detect_data, new Rect(4 + 80, i, 32, 1));//detect_data.Row(i).ColRange(4 + categ_nums, categ_nums + 36);

                    float cx = detect_data.At<float>(i, 0);
                    float cy = detect_data.At<float>(i, 1);
                    float ow = detect_data.At<float>(i, 2);
                    float oh = detect_data.At<float>(i, 3);
                    int x = (int)((cx - 0.5 * ow) * factor);
                    int y = (int)((cy - 0.5 * oh) * factor);
                    int width = (int)(ow * factor);
                    int height = (int)(oh * factor);
                    Rect box = new Rect();
                    box.X = x;
                    box.Y = y;
                    box.Width = width;
                    box.Height = height;

                    position_boxes.Add(box);
                    class_ids.Add(max_classId_point.X);
                    confidences.Add((float)max_score);
                    masks.Add(mask);
                }
            }


            int[] indexes = new int[position_boxes.Count];
            CvDnn.NMSBoxes(position_boxes, confidences, 0.5f, 0.5f, out indexes);

            SegResult result = new SegResult();
            Mat rgb_mask = Mat.Zeros(new Size((int)image.Size().Width, (int)image.Size().Height), MatType.CV_8UC3);
            Random rd = new Random(); // Generate Random Numbers
            for (int i = 0; i < indexes.Length; i++)
            {
                int index = indexes[i];
                // Division scope
                Rect box = position_boxes[index];
                int box_x1 = Math.Max(0, box.X);
                int box_y1 = Math.Max(0, box.Y);
                int box_x2 = Math.Max(0, box.BottomRight.X);
                int box_y2 = Math.Max(0, box.BottomRight.Y);

                // Segmentation results
                Mat original_mask = masks[index] * proto_data;
                for (int col = 0; col < original_mask.Cols; col++)
                {
                    original_mask.Set<float>(0, col, sigmoid(original_mask.At<float>(0, col)));
                }
                // 1x25600 -> 160x160 Convert to original size
                Mat reshape_mask = original_mask.Reshape(1, 160);

                //Console.WriteLine("m1.size = {0}", m1.Size());

                // Split size after scaling
                int mx1 = Math.Max(0, (int)((box_x1 / factor) * 0.25));
                int mx2 = Math.Min(160, (int)((box_x2 / factor) * 0.25));
                int my1 = Math.Max(0, (int)((box_y1 / factor) * 0.25));
                int my2 = Math.Min(160, (int)((box_y2 / factor) * 0.25));
                // Crop Split Region
                Mat mask_roi = new Mat(reshape_mask, new OpenCvSharp.Range(my1, my2), new OpenCvSharp.Range(mx1, mx2));
                // Convert the segmented area to the actual size of the image
                Mat actual_maskm = new Mat();
                Cv2.Resize(mask_roi, actual_maskm, new Size(box_x2 - box_x1, box_y2 - box_y1));
                // Binary segmentation region
                for (int r = 0; r < actual_maskm.Rows; r++)
                {
                    for (int c = 0; c < actual_maskm.Cols; c++)
                    {
                        float pv = actual_maskm.At<float>(r, c);
                        if (pv > 0.5)
                        {
                            actual_maskm.Set<float>(r, c, 1.0f);
                        }
                        else
                        {
                            actual_maskm.Set<float>(r, c, 0.0f);
                        }
                    }
                }

                // 预测
                Mat bin_mask = new Mat();
                actual_maskm = actual_maskm * 200;
                actual_maskm.ConvertTo(bin_mask, MatType.CV_8UC1);
                if ((box_y1 + bin_mask.Rows) >= (int)image.Size().Height)
                {
                    box_y2 = (int)image.Size().Height - 1;
                }
                if ((box_x1 + bin_mask.Cols) >= (int)image.Size().Width)
                {
                    box_x2 = (int)image.Size().Width - 1;
                }
                // Obtain segmentation area
                Mat mask = Mat.Zeros(new Size((int)image.Size().Width, (int)image.Size().Height), MatType.CV_8UC1);
                bin_mask = new Mat(bin_mask, new OpenCvSharp.Range(0, box_y2 - box_y1), new OpenCvSharp.Range(0, box_x2 - box_x1));
                Rect roi1 = new Rect(box_x1, box_y1, box_x2 - box_x1, box_y2 - box_y1);
                bin_mask.CopyTo(new Mat(mask, roi1));
                // Color segmentation area
                Cv2.Add(rgb_mask, new Scalar(rd.Next(0, 255), rd.Next(0, 255), rd.Next(0, 255)), rgb_mask, mask);
                result.add(class_ids[index], confidences[index], position_boxes[index], rgb_mask.Clone());
            }

            Mat masked_img = new Mat();
            // Draw recognition results on the image
            for (int i = 0; i < result.count; i++)
            {
                Cv2.Rectangle(image, result.datas[i].box, new Scalar(0, 0, 255), 2, LineTypes.Link8);
                Cv2.Rectangle(image, new Point(result.datas[i].box.TopLeft.X, result.datas[i].box.TopLeft.Y + 30),
                    new Point(result.datas[i].box.BottomRight.X, result.datas[i].box.TopLeft.Y), new Scalar(0, 255, 255), -1);
                Cv2.PutText(image, CocoOption.lables[result.datas[i].index] + "-" + result.datas[i].score.ToString("0.00"),
                    new Point(result.datas[i].box.X, result.datas[i].box.Y + 25),
                    HersheyFonts.HersheySimplex, 0.8, new Scalar(0, 0, 0), 2);
                Cv2.AddWeighted(image, 0.5, result.datas[i].mask, 0.5, 0, masked_img);
            }


            string output_path = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(image_path)),
                Path.GetFileNameWithoutExtension(image_path) + "_result.jpg");
            Cv2.ImWrite(output_path, masked_img);
            Slog.INFO("The result save to " + output_path);
            Cv2.ImShow("Result", masked_img);
            Cv2.WaitKey(0);

        }

        private static float sigmoid(float a)
        {
            float b = 1.0f / (1.0f + (float)Math.Exp(-a));
            return b;
        }
    }
}
