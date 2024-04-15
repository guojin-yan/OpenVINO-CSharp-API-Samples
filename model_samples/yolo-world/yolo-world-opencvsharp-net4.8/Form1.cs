using OpenCvSharp;
using OpenCvSharp.ImgHash;
using OpenVinoSharp;
using OpenVinoSharp.Extensions.process;
using OpenVinoSharp.Extensions.result;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Web.UI.WebControls;
using System.Windows.Forms;
using static OpenCvSharp.FileStorage;

namespace yolo_world_opencvsharp_net4._8
{
    public partial class Form1 : Form
    {
        Core core = null;
        Model model = null;
        CompiledModel compiled_model = null;
        InferRequest request = null;
        DateTime start = DateTime.Now;
        DateTime end = DateTime.Now;
        List<string> classes = null;
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

            cb_device.Items.Add("AUTO");
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
            //dlg.InitialDirectory = @"E:\Git_space\Csharp_deploy_Yolov8\model";
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
            //指定当前目录
            //dlg.InitialDirectory = System.Environment.CurrentDirectory;
            //dlg.InitialDirectory = System.IO.Path.GetFullPath(@"..//..//..//..");
            //设置文件过滤效果
            dlg.Filter = "图片文件(*.png,*.jpg,*.jepg)|*.png;*.jpg;*.jepg";
            //dlg.InitialDirectory = @"E:\Git_space\Csharp_deploy_Yolov8\demo";
            //判断文件对话框是否打开
            if (dlg.ShowDialog() == DialogResult.OK)
            {
                tb_input_path.Text = dlg.FileName;
            }
        }

        private void btn_load_model_Click(object sender, EventArgs e)
        {
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
            Tensor input_tensor = request.get_input_tensor();
            Shape input_shape = input_tensor.get_shape();
            float factor = 0f;
            Mat image = Cv2.ImRead(tb_input_path.Text);

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
            start = DateTime.Now;

            Tensor output_boxes = request.get_tensor("boxes");
            float[] boxes = output_boxes.get_data<float>((int)output_boxes.get_size());
            Tensor output_scores = request.get_tensor("scores");
            float[] scores = output_scores.get_data<float>((int)output_scores.get_size());
            Tensor output_labels = request.get_tensor("labels");
            int[] labels = output_labels.get_data<int>((int)output_labels.get_size());

            Mat result_mat = draw_bbox(image.Clone(), boxes, scores, labels, factor);
            end = DateTime.Now;
            tb_msg.AppendText("Process result data: " + (end - start).TotalMilliseconds + "ms.\r\n");
            start = DateTime.Now;
            pictureBox2.BackgroundImage = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(result_mat);
        }

        Mat draw_bbox(Mat image, float[] boxes, float[] scores, int[] labels, float factor)
        {
            for (int i = 0; i < scores.Length; i++)
            {
                if (scores[i] < 0.2f)
                {
                    continue;
                }
                float tx = boxes[4 * i];
                float ty = boxes[4 * i + 1];
                float bx = boxes[4 * i + 2];
                float by = boxes[4 * i + 3];
                int x = (int)(tx * factor);
                int y = (int)(ty * factor);
                int width = (int)((bx - tx) * factor);
                int height = (int)((by - ty) * factor);
                Rect box = new Rect();
                box.X = x;
                box.Y = y;
                box.Width = width;
                box.Height = height;
                Cv2.Rectangle(image, box, new Scalar(0, 0, 255),2);
                Cv2.PutText(image, classes[labels[i]] + "-" + scores[i].ToString("0.00"), 
                    new OpenCvSharp.Point(x, y - 10), HersheyFonts.HersheySimplex, 1, new Scalar(255, 0, 0), 2);
            }
            return image;
        }
    }
}
