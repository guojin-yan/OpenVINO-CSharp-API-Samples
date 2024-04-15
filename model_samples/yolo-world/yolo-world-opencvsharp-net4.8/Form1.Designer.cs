namespace yolo_world_opencvsharp_net4._8
{
    partial class Form1
    {
        /// <summary>
        /// 必需的设计器变量。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 清理所有正在使用的资源。
        /// </summary>
        /// <param name="disposing">如果应释放托管资源，为 true；否则为 false。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows 窗体设计器生成的代码

        /// <summary>
        /// 设计器支持所需的方法 - 不要修改
        /// 使用代码编辑器修改此方法的内容。
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form1));
            this.tb_classes = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.label3 = new System.Windows.Forms.Label();
            this.btn_select_model = new System.Windows.Forms.Button();
            this.tb_model_path = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.btn_select_input = new System.Windows.Forms.Button();
            this.btn_infer = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.tb_input_path = new System.Windows.Forms.TextBox();
            this.btn_load_model = new System.Windows.Forms.Button();
            this.cb_device = new System.Windows.Forms.ComboBox();
            this.label5 = new System.Windows.Forms.Label();
            this.pictureBox2 = new System.Windows.Forms.PictureBox();
            this.tb_msg = new System.Windows.Forms.TextBox();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).BeginInit();
            this.SuspendLayout();
            // 
            // tb_classes
            // 
            this.tb_classes.Font = new System.Drawing.Font("宋体", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.tb_classes.Location = new System.Drawing.Point(36, 128);
            this.tb_classes.Multiline = true;
            this.tb_classes.Name = "tb_classes";
            this.tb_classes.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.tb_classes.Size = new System.Drawing.Size(491, 184);
            this.tb_classes.TabIndex = 0;
            this.tb_classes.Text = resources.GetString("tb_classes.Text");
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("宋体", 13.8F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.label1.Location = new System.Drawing.Point(32, 82);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(322, 23);
            this.label1.TabIndex = 1;
            this.label1.Text = "输入要检测的类，用逗号分隔";
            // 
            // pictureBox1
            // 
            this.pictureBox1.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Zoom;
            this.pictureBox1.Location = new System.Drawing.Point(549, 66);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(1000, 500);
            this.pictureBox1.TabIndex = 2;
            this.pictureBox1.TabStop = false;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Font = new System.Drawing.Font("黑体", 18F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.label3.Location = new System.Drawing.Point(522, 23);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(561, 30);
            this.label3.TabIndex = 1;
            this.label3.Text = "YOLO-World 实时开放词汇对象检测演示";
            // 
            // btn_select_model
            // 
            this.btn_select_model.Font = new System.Drawing.Font("宋体", 13.8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.btn_select_model.Location = new System.Drawing.Point(68, 513);
            this.btn_select_model.Name = "btn_select_model";
            this.btn_select_model.Size = new System.Drawing.Size(157, 36);
            this.btn_select_model.TabIndex = 3;
            this.btn_select_model.Text = "选择模型";
            this.btn_select_model.UseVisualStyleBackColor = true;
            this.btn_select_model.Click += new System.EventHandler(this.btn_select_model_Click);
            // 
            // tb_model_path
            // 
            this.tb_model_path.Font = new System.Drawing.Font("宋体", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.tb_model_path.Location = new System.Drawing.Point(157, 354);
            this.tb_model_path.Name = "tb_model_path";
            this.tb_model_path.Size = new System.Drawing.Size(370, 30);
            this.tb_model_path.TabIndex = 4;
            this.tb_model_path.Text = "E:\\Model\\yolo-world\\yolow-l.onnx";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Font = new System.Drawing.Font("宋体", 13.8F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.label4.Location = new System.Drawing.Point(32, 357);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(119, 23);
            this.label4.TabIndex = 1;
            this.label4.Text = "模型地址:";
            // 
            // btn_select_input
            // 
            this.btn_select_input.Font = new System.Drawing.Font("宋体", 13.8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.btn_select_input.Location = new System.Drawing.Point(296, 513);
            this.btn_select_input.Name = "btn_select_input";
            this.btn_select_input.Size = new System.Drawing.Size(157, 36);
            this.btn_select_input.TabIndex = 3;
            this.btn_select_input.Text = "选择输入";
            this.btn_select_input.UseVisualStyleBackColor = true;
            this.btn_select_input.Click += new System.EventHandler(this.btn_select_input_Click);
            // 
            // btn_infer
            // 
            this.btn_infer.Font = new System.Drawing.Font("宋体", 15F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.btn_infer.Location = new System.Drawing.Point(287, 581);
            this.btn_infer.Name = "btn_infer";
            this.btn_infer.Size = new System.Drawing.Size(221, 48);
            this.btn_infer.TabIndex = 3;
            this.btn_infer.Text = "模 型 推 理";
            this.btn_infer.UseVisualStyleBackColor = true;
            this.btn_infer.Click += new System.EventHandler(this.btn_infer_Click);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("宋体", 13.8F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.label2.Location = new System.Drawing.Point(32, 400);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(119, 23);
            this.label2.TabIndex = 1;
            this.label2.Text = "输入地址:";
            // 
            // tb_input_path
            // 
            this.tb_input_path.Font = new System.Drawing.Font("宋体", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.tb_input_path.Location = new System.Drawing.Point(157, 397);
            this.tb_input_path.Name = "tb_input_path";
            this.tb_input_path.Size = new System.Drawing.Size(370, 30);
            this.tb_input_path.TabIndex = 4;
            this.tb_input_path.Text = "E:\\ModelData\\image\\demo_2.jpg";
            // 
            // btn_load_model
            // 
            this.btn_load_model.Font = new System.Drawing.Font("宋体", 15F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.btn_load_model.Location = new System.Drawing.Point(36, 581);
            this.btn_load_model.Name = "btn_load_model";
            this.btn_load_model.Size = new System.Drawing.Size(221, 48);
            this.btn_load_model.TabIndex = 3;
            this.btn_load_model.Text = "加 载 模 型";
            this.btn_load_model.UseVisualStyleBackColor = true;
            this.btn_load_model.Click += new System.EventHandler(this.btn_load_model_Click);
            // 
            // cb_device
            // 
            this.cb_device.Font = new System.Drawing.Font("宋体", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.cb_device.FormattingEnabled = true;
            this.cb_device.Location = new System.Drawing.Point(157, 446);
            this.cb_device.Name = "cb_device";
            this.cb_device.Size = new System.Drawing.Size(205, 28);
            this.cb_device.TabIndex = 5;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Font = new System.Drawing.Font("宋体", 13.8F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.label5.Location = new System.Drawing.Point(32, 449);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(119, 23);
            this.label5.TabIndex = 1;
            this.label5.Text = "输入地址:";
            // 
            // pictureBox2
            // 
            this.pictureBox2.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Zoom;
            this.pictureBox2.Location = new System.Drawing.Point(549, 591);
            this.pictureBox2.Name = "pictureBox2";
            this.pictureBox2.Size = new System.Drawing.Size(1000, 500);
            this.pictureBox2.TabIndex = 2;
            this.pictureBox2.TabStop = false;
            // 
            // tb_msg
            // 
            this.tb_msg.Font = new System.Drawing.Font("宋体", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.tb_msg.Location = new System.Drawing.Point(36, 658);
            this.tb_msg.Multiline = true;
            this.tb_msg.Name = "tb_msg";
            this.tb_msg.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.tb_msg.Size = new System.Drawing.Size(490, 418);
            this.tb_msg.TabIndex = 0;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1582, 1103);
            this.Controls.Add(this.cb_device);
            this.Controls.Add(this.tb_input_path);
            this.Controls.Add(this.tb_model_path);
            this.Controls.Add(this.btn_load_model);
            this.Controls.Add(this.btn_infer);
            this.Controls.Add(this.btn_select_input);
            this.Controls.Add(this.btn_select_model);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.pictureBox2);
            this.Controls.Add(this.pictureBox1);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.tb_msg);
            this.Controls.Add(this.tb_classes);
            this.Name = "Form1";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Form1";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TextBox tb_classes;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Button btn_select_model;
        private System.Windows.Forms.TextBox tb_model_path;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Button btn_select_input;
        private System.Windows.Forms.Button btn_infer;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox tb_input_path;
        private System.Windows.Forms.Button btn_load_model;
        private System.Windows.Forms.ComboBox cb_device;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.PictureBox pictureBox2;
        private System.Windows.Forms.TextBox tb_msg;
    }
}

