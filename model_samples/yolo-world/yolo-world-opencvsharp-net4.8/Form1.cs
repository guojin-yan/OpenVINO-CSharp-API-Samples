using OpenCvSharp;
using OpenCvSharp.Dnn;
using OpenVinoSharp;
using OpenVinoSharp.Extensions.model;
using OpenVinoSharp.Extensions.process;
using OpenVinoSharp.Extensions.result;
using System;
using System.Collections.Generic;
using System.IO;
using System.Net.NetworkInformation;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows.Forms;

namespace yolo_world_opencvsharp_net4._8
{
    public partial class Form1 : Form
    {
        public Core core = null;
        public Model model = null;
        public CompiledModel compiled_model = null;
        public InferRequest request = null;
        DateTime start = DateTime.Now;
        DateTime end = DateTime.Now;
        public List<string> classes = null;
        public Form1()
        {
            InitializeComponent();
        }
        private void Form1_Load(object sender, EventArgs e)
        {
            start = DateTime.Now;
            core = new Core();
            end = DateTime.Now;
            tb_msg.AppendText("Initialize OpenVINO Runtime Core: " + (end - start).TotalMilliseconds + "ms.\r\n");
            List<string> devices = core.get_available_devices();
            foreach (var item in devices)
            {
                cb_device.Items.Add(item);
            }
            cb_device.SelectedIndex = 0;
        }

        private void btn_select_model_Click(object sender, EventArgs e)
        {
            OpenFileDialog dlg = new OpenFileDialog();
            //若要改变对话框标题
            dlg.Title = "选择推理模型文件";
            //设置文件过滤效果
            dlg.Filter = "模型文件(*.pdmodel,*.onnx,*.xml)|*.pdmodel;*.onnx;*.xml";
            //判断文件对话框是否打开
            if (dlg.ShowDialog() == DialogResult.OK)
            {
                tb_model_path.Text = dlg.FileName;
            }
        }

        private void btn_select_input_Click(object sender, EventArgs e)
        {
            OpenFileDialog dlg = new OpenFileDialog();
            //若要改变对话框标题
            dlg.Title = "选择测试输入文件";
            //设置文件过滤效果
            dlg.Filter = "图片文件(*.png,*.jpg,*.jepg,*.mp4)|*.png;*.jpg;*.jepg;*.mp4";
            //判断文件对话框是否打开
            if (dlg.ShowDialog() == DialogResult.OK)
            {
                tb_input_path.Text = dlg.FileName;
            }
        }


        private void btn_load_model_Click(object sender, EventArgs e)
        {
            string[] lines = tb_msg.Text.Split(new string[] { "\r\n" }, StringSplitOptions.None);
            tb_msg.Clear();
            tb_msg.Text = lines[0] + "\r\n";
            start = DateTime.Now;
            model = core.read_model(tb_model_path.Text);
            end = DateTime.Now;
            tb_msg.AppendText("Read inference model: " + (end - start).TotalMilliseconds + "ms.\r\n");
            start = DateTime.Now;
            compiled_model = core.compile_model(model, cb_device.SelectedItem.ToString());
            end = DateTime.Now;
            tb_msg.AppendText("Loading a model to the device: " + (end - start).TotalMilliseconds + "ms.\r\n");
            start = DateTime.Now;
            request = compiled_model.create_infer_request();
            end = DateTime.Now;
            tb_msg.AppendText("Create an infer request: " + (end - start).TotalMilliseconds + "ms.\r\n");
        }
        private void btn_infer_Click(object sender, EventArgs e)
        {
            string[] words = tb_classes.Text.Split(',');
            classes = new List<string>(words);
            if (Path.GetExtension(tb_input_path.Text) == ".mp4")
            {
                VideoCapture video = new VideoCapture(tb_input_path.Text);
                if (video.IsOpened()) 
                {
                    Mat frame = new Mat();
                    video.Read(frame);
                    while (!frame.Empty())
                    {
                        image_predict(frame);
                        video.Read(frame);
                        Thread.Sleep(10);
                    }
                }
            }
            else 
            { 
                Mat image = Cv2.ImRead(tb_input_path.Text); 
                image_predict(image); 
            }
           
     
           
            
        }
        void image_predict(Mat image) 
        {
            Tensor input_tensor = request.get_input_tensor();
            Shape input_shape = input_tensor.get_shape();
            float factor = 0f;
            pictureBox1.BackgroundImage = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(image);

            start = DateTime.Now;
            Mat mat = new Mat();
            Cv2.CvtColor(image, mat, ColorConversionCodes.BGR2RGB);
            mat = OpenVinoSharp.Extensions.process.Resize.letterbox_img(mat, (int)input_shape[2], out factor);
            mat = Normalize.run(mat, true);
            float[] input_data = Permute.run(mat);
            input_tensor.set_data(input_data);
            end = DateTime.Now;
            tb_msg.AppendText("Process input images: " + (end - start).TotalMilliseconds + "ms.\r\n");
            start = DateTime.Now;
            request.infer();
            end = DateTime.Now;
            tb_msg.AppendText("Do inference synchronously: " + (end - start).TotalMilliseconds + "ms.\r\n");
            float fps = (float)(1000.0f / ((end - start).TotalMilliseconds));
            start = DateTime.Now;

            Tensor output_tensor = request.get_output_tensor();

            Shape output_shape = output_tensor.get_shape();

            int categ_nums = (int)output_shape[1] - 4;
            DetResult result = postprocess(output_tensor.get_data<float>((int)output_tensor.get_size()), categ_nums, factor);

            Mat result_mat = image.Clone();
            for (int i = 0; i < result.count; i++)
            {
                Cv2.Rectangle(result_mat, result.datas[i].box, new Scalar(0.0, 0.0, 255.0), 2);
                Cv2.Rectangle(result_mat, new Point(result.datas[i].box.TopLeft.X, result.datas[i].box.TopLeft.Y + 30), new Point(result.datas[i].box.BottomRight.X, result.datas[i].box.TopLeft.Y), new Scalar(0.0, 255.0, 255.0), -1);
                Cv2.PutText(result_mat, classes[result.datas[i].index] + "-" + result.datas[i].score.ToString("0.00"), new Point(result.datas[i].box.X, result.datas[i].box.Y + 25), HersheyFonts.HersheySimplex, 0.8, new Scalar(0.0, 0.0, 0.0), 2);
            }
            end = DateTime.Now;
            tb_msg.AppendText("Process result data: " + (end - start).TotalMilliseconds + "ms.\r\n");
            start = DateTime.Now;
            Cv2.Rectangle(result_mat, new Point(30,20), new Point(250,60), new Scalar(0.0, 255.0, 255.0), -1);
            Cv2.PutText(result_mat, "FPS: " + fps.ToString("0.00"), new Point(50, 50), HersheyFonts.HersheySimplex, 0.8, new Scalar(0, 0, 0), 2);
            Cv2.WaitKey(1);
            pictureBox2.BackgroundImage = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(result_mat);
        }


        DetResult postprocess(float[] result, int categ_nums, float factor) 
        {
            Mat result_data = new Mat(4 + categ_nums, 8400, MatType.CV_32F,result);
            result_data = result_data.T();

            // Storage results list
            List<Rect> position_boxes = new List<Rect>();
            List<int> classIds = new List<int>();
            List<float> confidences = new List<float>();
            // Preprocessing output results
            for (int i = 0; i < result_data.Rows; i++)
            {
                Mat classesScores = new Mat(result_data, new Rect(4, i, categ_nums, 1));
                Point maxClassIdPoint, minClassIdPoint;
                double maxScore, minScore;
                // Obtain the maximum value and its position in a set of data
                Cv2.MinMaxLoc(classesScores, out minScore, out maxScore,
                    out minClassIdPoint, out maxClassIdPoint);
                // Confidence level between 0 ~ 1
                // Obtain identification box information
                if (maxScore > 0.25)
                {
                    float cx = result_data.At<float>(i, 0);
                    float cy = result_data.At<float>(i, 1);
                    float ow = result_data.At<float>(i, 2);
                    float oh = result_data.At<float>(i, 3);
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
                    classIds.Add(maxClassIdPoint.X);
                    confidences.Add((float)maxScore);
                }
            }
            // NMS non maximum suppression
            int[] indexes = new int[position_boxes.Count];
            float score = float.Parse(tb_score.Text);
            float nms = float.Parse(tb_nms.Text);
            CvDnn.NMSBoxes(position_boxes, confidences, score, nms, out indexes);
            DetResult re = new DetResult();
            // 
            for (int i = 0; i < indexes.Length; i++)
            {
                int index = indexes[i];
                re.add(classIds[index], confidences[index], position_boxes[index]);
            }
            return re;
        }

    }
}
