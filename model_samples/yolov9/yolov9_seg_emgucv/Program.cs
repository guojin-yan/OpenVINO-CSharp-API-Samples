using OpenVinoSharp.Extensions.utility;
using OpenVinoSharp;
using Emgu.CV.CvEnum;
using Emgu.CV;
using OpenVinoSharp.Extensions.model;
using OpenVinoSharp.Extensions.result;
using OpenVinoSharp.Extensions;
using System.Drawing;
using System.Runtime.InteropServices;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using Emgu.CV.Linemod;
using OpenVinoSharp.preprocess;

namespace yolov9_seg_emgucv
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
                    _ = Download.download_file_async("https://github.com/guojin-yan/OpenVINO-CSharp-API-Samples/releases/download/Image/test_det_01.jpg",
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
            Mat max_image = Mat.Zeros(max_image_length, max_image_length, DepthType.Cv8U,3);
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

            Tensor output_tensor_0 = infer_request.get_output_tensor(0);
            float[] result_detect = output_tensor_0.get_data<float>((int)output_tensor_0.get_size());

            Tensor output_tensor_1 = infer_request.get_output_tensor(1);
            float[] result_proto = output_tensor_1.get_data<float>((int)output_tensor_1.get_size());


            Mat detect_data = new Mat(116, 8400, DepthType.Cv32F, 1,
                  Marshal.UnsafeAddrOfPinnedArrayElement(result_detect, 0), 4 * 8400);
            Mat proto_data = new Mat(32, 25600, DepthType.Cv32F, 1,
                Marshal.UnsafeAddrOfPinnedArrayElement(result_proto, 0), 4*25600);
            detect_data = detect_data.T();
            List<Rectangle> position_boxes = new List<Rectangle>();
            List<int> class_ids = new List<int>();
            List<float> confidences = new List<float>();
            List<Mat> masks = new List<Mat>();
            for (int i = 0; i < detect_data.Rows; i++)
            {

                Mat classes_scores = new Mat(detect_data, new Rectangle(4, i, 80, 1));// GetArray(i, 5, classes_scores);
                Point max_classId_point = new Point(), min_classId_point = new Point();
                double max_score = 0, min_score = 0;
                CvInvoke.MinMaxLoc(classes_scores, ref min_score, ref max_score,
                    ref min_classId_point, ref max_classId_point);

                if (max_score > 0.25)
                {
                    //Console.WriteLine(max_score);

                    Mat mask = new Mat(detect_data, new Rectangle(4 + 80, i, 32, 1));//detect_data.Row(i).ColRange(4 + categ_nums, categ_nums + 36);

                    Mat mat = new Mat(detect_data, new Rectangle(0, i, 4, 1));
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
                    masks.Add(mask);
                }
            }


            int[] indexes = DnnInvoke.NMSBoxes(position_boxes.ToArray(), confidences.ToArray(), 0.5f, 0.5f);

            SegResult result = new SegResult(); // Output Result Class
                                            // RGB images with colors
            Mat rgb_mask = Mat.Zeros((int)image.Size.Height, (int)image.Size.Width, DepthType.Cv8U, 3);
            Random rd = new Random(); // Generate Random Numbers

            Matrix<float> proto_data_m = new Matrix<float>(proto_data.Rows, proto_data.Cols);
            proto_data.CopyTo(proto_data_m);
            for (int i = 0; i < indexes.Length; i++)
            {
                int index = indexes[i];
                // Division scope
                Rectangle box = position_boxes[index];
                int box_x1 = Math.Max(0, box.X);
                int box_y1 = Math.Max(0, box.Y);
                int box_x2 = Math.Max(0, box.X + box.Width);
                int box_y2 = Math.Max(0, box.Y + box.Height);

                // Segmentation results
                //Mat original_mask = new Mat(32,32, DepthType.Cv32F, 1);


                Matrix<float> xx = new Matrix<float>(masks[index].Rows, masks[index].Cols);
                masks[index].CopyTo(xx);

                Matrix<float> rr = xx * proto_data_m;
                Mat original_mask = rr.Mat;
                float[,] data = (float[,])original_mask.GetData();
                float[] data2 = new float[data.GetLength(1)];
                for (int col = 0; col < original_mask.Cols; col++)
                {
                    data2[col] = sigmoid(data[0, col]);
                }
                Mat original_mask1 = new Mat(original_mask.Size.Height, original_mask.Size.Width, DepthType.Cv32F, 1,
                    Marshal.UnsafeAddrOfPinnedArrayElement(data2, 0), 4 * original_mask.Cols);
                // 1x25600 -> 160x160 Convert to original size
                Mat reshape_mask = original_mask1.Reshape(1, 160);

                //Console.WriteLine("m1.size = {0}", m1.Size());

                // Split size after scaling
                int mx1 = Math.Max(0, (int)((box_x1 / factor) * 0.25));
                int mx2 = Math.Min(160, (int)((box_x2 / factor) * 0.25));
                int my1 = Math.Max(0, (int)((box_y1 / factor) * 0.25));
                int my2 = Math.Min(160, (int)((box_y2 / factor) * 0.25));
                // Crop Split Region
                Mat mask_roi = new Mat(reshape_mask, new Emgu.CV.Structure.Range(my1, my2), new Emgu.CV.Structure.Range(mx1, mx2));
                // Convert the segmented area to the actual size of the image
                Mat actual_maskm = new Mat();
                CvInvoke.Resize(mask_roi, actual_maskm, new Size(box_x2 - box_x1, box_y2 - box_y1));
                // Binary segmentation region
                float[,] data1 = (float[,])actual_maskm.GetData();
                for (int r = 0; r < actual_maskm.Rows; r++)
                {
                    for (int c = 0; c < actual_maskm.Cols; c++)
                    {
                        float pv = data1[r, c];
                        if (pv > 0.5)
                        {
                            data1[r, c] = 1.0f;
                        }
                        else
                        {
                            data1[r, c] = 0.0f;
                        }
                    }
                }
                actual_maskm = new Mat(actual_maskm.Size, DepthType.Cv32F, 1,
                    Marshal.UnsafeAddrOfPinnedArrayElement(data1, 0), 4 * actual_maskm.Cols);
                // 预测
                Mat bin_mask = new Mat();
                actual_maskm = actual_maskm * 200;
                actual_maskm.ConvertTo(bin_mask, DepthType.Cv8U);
                if ((box_y1 + bin_mask.Rows) >= (int)image.Size.Height)
                {
                    box_y2 = (int)image.Size.Height - 1;
                }
                if ((box_x1 + bin_mask.Cols) >= (int)image.Size.Width)
                {
                    box_x2 = (int)image.Size.Width - 1;
                }
                // Obtain segmentation area
                Mat mask = Mat.Zeros((int)image.Size.Height, (int)image.Size.Width, DepthType.Cv8U, 1);
                bin_mask = new Mat(bin_mask, new Rectangle(0, 0, box_x2 - box_x1, box_y2 - box_y1));
                Rectangle roi1 = new Rectangle(box_x1, box_y1, box_x2 - box_x1, box_y2 - box_y1);
                bin_mask.CopyTo(new Mat(mask, roi1));
                Mat new_rgb_mask = new Mat();
                // Color segmentation area
                CvInvoke.Add(rgb_mask, new ScalarArray(new MCvScalar(rd.Next(0, 255), rd.Next(0, 255), rd.Next(0, 255))), new_rgb_mask, mask);
                //CvInvoke.Imshow("new_rgb_mask", new_rgb_mask);
                //CvInvoke.WaitKey(0);
                result.add(class_ids[index], confidences[index], position_boxes[index], new_rgb_mask.Clone());

            }



            Mat masked_img = new Mat(image.Size, DepthType.Cv8U, 3);
            // Draw recognition results on the image
            for (int i = 0; i < result.count; i++)
            {
                CvInvoke.Rectangle(image, result.datas[i].box, new MCvScalar(0, 0, 255), 2, LineType.Filled);
                CvInvoke.Rectangle(image, new Rectangle(new Point(result.datas[i].box.X, result.datas[i].box.Y),
                 new Size(result.datas[i].box.Width, 30)), new MCvScalar(0, 255, 255), -1);
                CvInvoke.PutText(image, CocoOption.lables[result.datas[i].index] + "-" + result.datas[i].score.ToString("0.00"),
                    new Point(result.datas[i].box.X, result.datas[i].box.Y + 25),
                    FontFace.HersheySimplex, 0.8, new MCvScalar(0, 0, 0), 2);
                CvInvoke.AddWeighted(masked_img, 1, result.datas[i].mask, 1, 0, masked_img);
            }
            CvInvoke.AddWeighted(masked_img, 0.5, image, 0.5, 0, masked_img);

            string output_path = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(image_path)),
                Path.GetFileNameWithoutExtension(image_path) + "_result.jpg");
            CvInvoke.Imwrite(output_path, masked_img);
            Slog.INFO("The result save to " + output_path);
            CvInvoke.Imshow("Result", masked_img);
            CvInvoke.WaitKey(0);

        }

        static void yolov9_seg_with_process(string model_path, string image_path, string device)
        {
            // -------- Step 1. Initialize OpenVINO Runtime Core --------
            DateTime start = DateTime.Now;
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
            byte[] input_data = new byte[input_shape[1] * input_shape[2] * input_shape[3]];
            //max_image.GetArray<int>(out input_data);
            max_image.CopyTo<byte>(input_data);
            input_tensor.set_data<byte>(input_data);

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


            Mat detect_data = new Mat(116, 8400, DepthType.Cv32F, 1,
                  Marshal.UnsafeAddrOfPinnedArrayElement(result_detect, 0), 4 * 8400);
            Mat proto_data = new Mat(32, 25600, DepthType.Cv32F, 1,
                Marshal.UnsafeAddrOfPinnedArrayElement(result_proto, 0), 4 * 25600);
            detect_data = detect_data.T();
            List<Rectangle> position_boxes = new List<Rectangle>();
            List<int> class_ids = new List<int>();
            List<float> confidences = new List<float>();
            List<Mat> masks = new List<Mat>();
            for (int i = 0; i < detect_data.Rows; i++)
            {

                Mat classes_scores = new Mat(detect_data, new Rectangle(4, i, 80, 1));// GetArray(i, 5, classes_scores);
                Point max_classId_point = new Point(), min_classId_point = new Point();
                double max_score = 0, min_score = 0;
                CvInvoke.MinMaxLoc(classes_scores, ref min_score, ref max_score,
                    ref min_classId_point, ref max_classId_point);

                if (max_score > 0.25)
                {
                    //Console.WriteLine(max_score);

                    Mat mask = new Mat(detect_data, new Rectangle(4 + 80, i, 32, 1));//detect_data.Row(i).ColRange(4 + categ_nums, categ_nums + 36);

                    Mat mat = new Mat(detect_data, new Rectangle(0, i, 4, 1));
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
                    masks.Add(mask);
                }
            }


            int[] indexes = DnnInvoke.NMSBoxes(position_boxes.ToArray(), confidences.ToArray(), 0.5f, 0.5f);

            SegResult result = new SegResult(); // Output Result Class
                                                // RGB images with colors
            Mat rgb_mask = Mat.Zeros((int)image.Size.Height, (int)image.Size.Width, DepthType.Cv8U, 3);
            Random rd = new Random(); // Generate Random Numbers

            Matrix<float> proto_data_m = new Matrix<float>(proto_data.Rows, proto_data.Cols);
            proto_data.CopyTo(proto_data_m);
            for (int i = 0; i < indexes.Length; i++)
            {
                int index = indexes[i];
                // Division scope
                Rectangle box = position_boxes[index];
                int box_x1 = Math.Max(0, box.X);
                int box_y1 = Math.Max(0, box.Y);
                int box_x2 = Math.Max(0, box.X + box.Width);
                int box_y2 = Math.Max(0, box.Y + box.Height);

                // Segmentation results
                //Mat original_mask = new Mat(32,32, DepthType.Cv32F, 1);


                Matrix<float> xx = new Matrix<float>(masks[index].Rows, masks[index].Cols);
                masks[index].CopyTo(xx);

                Matrix<float> rr = xx * proto_data_m;
                Mat original_mask = rr.Mat;
                float[,] data = (float[,])original_mask.GetData();
                float[] data2 = new float[data.GetLength(1)];
                for (int col = 0; col < original_mask.Cols; col++)
                {
                    data2[col] = sigmoid(data[0, col]);
                }
                Mat original_mask1 = new Mat(original_mask.Size.Height, original_mask.Size.Width, DepthType.Cv32F, 1,
                    Marshal.UnsafeAddrOfPinnedArrayElement(data2, 0), 4 * original_mask.Cols);
                // 1x25600 -> 160x160 Convert to original size
                Mat reshape_mask = original_mask1.Reshape(1, 160);

                //Console.WriteLine("m1.size = {0}", m1.Size());

                // Split size after scaling
                int mx1 = Math.Max(0, (int)((box_x1 / factor) * 0.25));
                int mx2 = Math.Min(160, (int)((box_x2 / factor) * 0.25));
                int my1 = Math.Max(0, (int)((box_y1 / factor) * 0.25));
                int my2 = Math.Min(160, (int)((box_y2 / factor) * 0.25));
                // Crop Split Region
                Mat mask_roi = new Mat(reshape_mask, new Emgu.CV.Structure.Range(my1, my2), new Emgu.CV.Structure.Range(mx1, mx2));
                // Convert the segmented area to the actual size of the image
                Mat actual_maskm = new Mat();
                CvInvoke.Resize(mask_roi, actual_maskm, new Size(box_x2 - box_x1, box_y2 - box_y1));
                // Binary segmentation region
                float[,] data1 = (float[,])actual_maskm.GetData();
                for (int r = 0; r < actual_maskm.Rows; r++)
                {
                    for (int c = 0; c < actual_maskm.Cols; c++)
                    {
                        float pv = data1[r, c];
                        if (pv > 0.5)
                        {
                            data1[r, c] = 1.0f;
                        }
                        else
                        {
                            data1[r, c] = 0.0f;
                        }
                    }
                }
                actual_maskm = new Mat(actual_maskm.Size, DepthType.Cv32F, 1,
                    Marshal.UnsafeAddrOfPinnedArrayElement(data1, 0), 4 * actual_maskm.Cols);
                // 预测
                Mat bin_mask = new Mat();
                actual_maskm = actual_maskm * 200;
                actual_maskm.ConvertTo(bin_mask, DepthType.Cv8U);
                if ((box_y1 + bin_mask.Rows) >= (int)image.Size.Height)
                {
                    box_y2 = (int)image.Size.Height - 1;
                }
                if ((box_x1 + bin_mask.Cols) >= (int)image.Size.Width)
                {
                    box_x2 = (int)image.Size.Width - 1;
                }
                // Obtain segmentation area
                Mat mask = Mat.Zeros((int)image.Size.Height, (int)image.Size.Width, DepthType.Cv8U, 1);
                bin_mask = new Mat(bin_mask, new Rectangle(0, 0, box_x2 - box_x1, box_y2 - box_y1));
                Rectangle roi1 = new Rectangle(box_x1, box_y1, box_x2 - box_x1, box_y2 - box_y1);
                bin_mask.CopyTo(new Mat(mask, roi1));
                Mat new_rgb_mask = new Mat();
                // Color segmentation area
                CvInvoke.Add(rgb_mask, new ScalarArray(new MCvScalar(rd.Next(0, 255), rd.Next(0, 255), rd.Next(0, 255))), new_rgb_mask, mask);
                //CvInvoke.Imshow("new_rgb_mask", new_rgb_mask);
                //CvInvoke.WaitKey(0);
                result.add(class_ids[index], confidences[index], position_boxes[index], new_rgb_mask.Clone());

            }



            Mat masked_img = new Mat(image.Size, DepthType.Cv8U, 3);
            // Draw recognition results on the image
            for (int i = 0; i < result.count; i++)
            {
                CvInvoke.Rectangle(image, result.datas[i].box, new MCvScalar(0, 0, 255), 2, LineType.Filled);
                CvInvoke.Rectangle(image, new Rectangle(new Point(result.datas[i].box.X, result.datas[i].box.Y),
                 new Size(result.datas[i].box.Width, 30)), new MCvScalar(0, 255, 255), -1);
                CvInvoke.PutText(image, CocoOption.lables[result.datas[i].index] + "-" + result.datas[i].score.ToString("0.00"),
                    new Point(result.datas[i].box.X, result.datas[i].box.Y + 25),
                    FontFace.HersheySimplex, 0.8, new MCvScalar(0, 0, 0), 2);
                CvInvoke.AddWeighted(masked_img, 1, result.datas[i].mask, 1, 0, masked_img);
            }
            CvInvoke.AddWeighted(masked_img, 0.5, image, 0.5, 0, masked_img);

            string output_path = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(image_path)),
                Path.GetFileNameWithoutExtension(image_path) + "_result.jpg");
            CvInvoke.Imwrite(output_path, masked_img);
            Slog.INFO("The result save to " + output_path);
            CvInvoke.Imshow("Result", masked_img);
            CvInvoke.WaitKey(0);

        }

        private static float sigmoid(float a)
        {
            float b = 1.0f / (1.0f + (float)Math.Exp(-a));
            return b;
        }
    }
}
