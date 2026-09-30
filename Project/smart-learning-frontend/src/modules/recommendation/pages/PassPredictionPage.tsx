import {
  Typography,
  Form,
  Select,
  Button,
  Card,
  Progress,
  Tag,
  Alert,
  Space,
  Row,
  Col,
  Spin,
  Divider,
  Flex,
} from "antd";
import {
  ExperimentOutlined,
  BarChartOutlined,
  CheckCircleFilled,
  CloseCircleFilled,
  BookOutlined,
  ClockCircleOutlined,
  FileExclamationOutlined,
  CalendarOutlined,
  SafetyCertificateOutlined,
} from "@ant-design/icons";
import { usePassPrediction } from "../hooks/useRecommendation";

const { Title, Text, Paragraph } = Typography;

const GPA_OPTIONS = [
  { value: "Dưới 2.0", label: "Dưới 2.0" },
  { value: "Từ 2.0 đến 2.49", label: "Từ 2.0 đến 2.49" },
  { value: "Từ 2.5 đến 3.19", label: "Từ 2.5 đến 3.19" },
  { value: "Từ 3.2 đến 3.59", label: "Từ 3.2 đến 3.59" },
  { value: "từ 3.6 trở lên", label: "Từ 3.6 trở lên" },
];

const STUDY_HOURS_OPTIONS = [
  { value: "Từ 0 đến 5 giờ", label: "Từ 0 đến 5 giờ" },
  { value: "Từ 6 đến10 giờ", label: "Từ 6 đến 10 giờ" },
  { value: "Từ 11đến 15 giờ", label: "Từ 11 đến 15 giờ" },
  { value: "Từ 16 đến 20 giờ", label: "Từ 16 đến 20 giờ" },
  { value: "Trên 20 giờ", label: "Trên 20 giờ" },
];

const FAILED_SUBJECTS_OPTIONS = [
  { value: "0", label: "0 môn" },
  { value: "1", label: "1 môn" },
  { value: "2", label: "2 môn" },
  { value: "3", label: "3 môn" },
  { value: "Từ 4 môn trở lên", label: "4 môn trở lên" },
];

const ATTENDANCE_OPTIONS = [
  { value: "< 50%", label: "Dưới 50%" },
  { value: "50–70%", label: "50–70%" },
  { value: "70–90%", label: "70–90%" },
  { value: ">90%", label: "Trên 90%" },
];

type RiskConfig = {
  color: string;
  bg: string;
  border: string;
  icon: React.ReactNode;
  title: string;
  description: string;
};

const RISK_CONFIG: Record<string, RiskConfig> = {
  LOW: {
    color: "#389e0d",
    bg: "#f6ffed",
    border: "#b7eb8f",
    icon: <SafetyCertificateOutlined />,
    title: "Rủi ro THẤP",
    description:
      "Bạn đang trên đà thành công! Hãy tiếp tục duy trì phong độ hiện tại.",
  },
  MEDIUM: {
    color: "#d48806",
    bg: "#fffbe6",
    border: "#ffe58f",
    icon: <ExperimentOutlined />,
    title: "Rủi ro TRUNG BÌNH",
    description:
      "Cần chú ý hơn đến việc học. Tăng thời gian tự học và đi học đầy đủ.",
  },
  HIGH: {
    color: "#cf1322",
    bg: "#fff2f0",
    border: "#ffa39e",
    icon: <CloseCircleFilled />,
    title: "Rủi ro CAO",
    description:
      "Cần cải thiện ngay! Liên hệ giảng viên cố vấn hoặc tham gia lớp học hỗ trợ.",
  },
};

const FORM_FIELDS = [
  {
    name: "gpa_bucket",
    label: "GPA tích lũy",
    icon: <BookOutlined style={{ color: "#004286" }} />,
    placeholder: "Chọn khoảng GPA của bạn",
    options: GPA_OPTIONS,
    rules: [{ required: true, message: "Vui lòng chọn khoảng GPA" }],
  },
  {
    name: "study_hours_bucket",
    label: "Số giờ tự học mỗi tuần",
    icon: <ClockCircleOutlined style={{ color: "#004286" }} />,
    placeholder: "Chọn số giờ tự học",
    options: STUDY_HOURS_OPTIONS,
    rules: [{ required: true, message: "Vui lòng chọn số giờ học" }],
  },
  {
    name: "failed_subjects_count",
    label: "Số môn đã trượt",
    icon: <FileExclamationOutlined style={{ color: "#004286" }} />,
    placeholder: "Chọn số môn đã trượt",
    options: FAILED_SUBJECTS_OPTIONS,
    rules: [{ required: true, message: "Vui lòng chọn số môn trượt" }],
  },
  {
    name: "attendance_bucket",
    label: "Tỷ lệ điểm danh",
    icon: <CalendarOutlined style={{ color: "#004286" }} />,
    placeholder: "Chọn tỷ lệ điểm danh",
    options: ATTENDANCE_OPTIONS,
    rules: [{ required: true, message: "Vui lòng chọn tỷ lệ điểm danh" }],
  },
];

export default function PassPredictionPage() {
  const [form] = Form.useForm();
  const { mutate, data, isPending, isError, error, reset } =
    usePassPrediction();

  const handleSubmit = (values: Record<string, string>) => {
    mutate(
      values as {
        gpa_bucket: string;
        study_hours_bucket: string;
        failed_subjects_count: string;
        attendance_bucket: string;
      },
    );
  };

  const result = data?.data ?? null;
  const passPercent = result ? Math.round(result.probability_pass * 100) : 0;
  const failPercent = result ? Math.round(result.probability_fail * 100) : 0;
  const risk = result
    ? (RISK_CONFIG[result.risk_level] ?? RISK_CONFIG.MEDIUM)
    : null;
  const isPass = result?.prediction_result === "PASS";

  return (
    <div
      style={{
        padding: "24px 0",
        minHeight: "calc(100vh - 120px)",
        background: "#EEF3FB",
      }}
    >
      <div style={{ maxWidth: 1400, margin: "0 auto", padding: "0 24px" }}>
        {/* Hero Banner */}
        <div
          style={{
            background:
              "linear-gradient(135deg, #004286 0%, #1a5fa8 50%, #004286 100%)",
            borderRadius: 16,
            padding: "36px 40px",
            marginBottom: 28,
            position: "relative",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              position: "absolute",
              top: "-50%",
              right: "-10%",
              width: 300,
              height: 300,
              background:
                "radial-gradient(circle, rgba(232,168,56,0.15) 0%, transparent 70%)",
              borderRadius: "50%",
            }}
          />
          <Flex align="center" gap={12} style={{ marginBottom: 6 }}>
            <BarChartOutlined style={{ fontSize: 26, color: "#F2C94C" }} />
            <Title
              level={2}
              style={{ margin: 0, color: "#fff", fontWeight: 700 }}
            >
              Dự Đoán Kết Quả Học Tập
            </Title>
          </Flex>
          <Paragraph
            style={{ margin: 0, fontSize: 15, color: "rgba(255,255,255,0.8)" }}
          >
            Điền thông tin học tập — mô hình AI sẽ phân tích và dự đoán khả năng
            vượt qua môn học.
          </Paragraph>
        </div>

        <Row gutter={[28, 28]}>
          {/* LEFT: Form */}
          <Col xs={24} lg={10}>
            <Card
              style={{
                borderRadius: 16,
                border: "none",
                boxShadow: "0 2px 12px rgba(26,60,110,0.08)",
              }}
            >
              <Flex align="center" gap={8} style={{ marginBottom: 20 }}>
                <ExperimentOutlined
                  style={{ color: "#004286", fontSize: 18 }}
                />
                <Text strong style={{ fontSize: 15, color: "#004286" }}>
                  Thông Tin Học Tập
                </Text>
              </Flex>
              <Form
                form={form}
                layout="vertical"
                onFinish={handleSubmit}
                onValuesChange={() => reset()}
                requiredMark={false}
              >
                {FORM_FIELDS.map((field) => (
                  <Form.Item
                    key={field.name}
                    label={
                      <Flex align="center" gap={6}>
                        {field.icon}
                        <span>{field.label}</span>
                      </Flex>
                    }
                    name={field.name}
                    rules={field.rules}
                  >
                    <Select
                      placeholder={field.placeholder}
                      options={field.options}
                      size="large"
                    />
                  </Form.Item>
                ))}
                <Form.Item style={{ marginBottom: 0, marginTop: 8 }}>
                  <Button
                    type="primary"
                    htmlType="submit"
                    size="large"
                    loading={isPending}
                    icon={<ExperimentOutlined />}
                    block
                  >
                    Dự Đoán Kết Quả
                  </Button>
                </Form.Item>
              </Form>
            </Card>
          </Col>

          {/* RIGHT: Result */}
          <Col xs={24} lg={14}>
            {isError && (
              <Alert
                type="error"
                message="Lỗi khi dự đoán"
                description={error?.message}
                showIcon
                closable
                style={{ marginBottom: 16, borderRadius: 10 }}
              />
            )}
            {isPending && (
              <Card
                style={{
                  borderRadius: 16,
                  border: "none",
                  boxShadow: "0 2px 12px rgba(26,60,110,0.08)",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    padding: "80px 0",
                  }}
                >
                  <Space direction="vertical" align="center">
                    <Spin size="large" />
                    <Text type="secondary">
                      Đang phân tích dữ liệu học tập...
                    </Text>
                  </Space>
                </div>
              </Card>
            )}
            {!isPending && result && (
              <Card
                style={{
                  borderRadius: 16,
                  border: "none",
                  boxShadow: "0 2px 12px rgba(26,60,110,0.08)",
                  borderTop: `3px solid ${isPass ? "#389e0d" : "#cf1322"}`,
                }}
              >
                <Flex
                  justify="space-between"
                  align="center"
                  style={{ marginBottom: 24 }}
                >
                  <Flex align="center" gap={10}>
                    {isPass ? (
                      <CheckCircleFilled
                        style={{ fontSize: 28, color: "#389e0d" }}
                      />
                    ) : (
                      <CloseCircleFilled
                        style={{ fontSize: 28, color: "#cf1322" }}
                      />
                    )}
                    <div>
                      <Text style={{ fontSize: 13, color: "#5a6a85" }}>
                        Kết Quả Dự Đoán
                      </Text>
                      <br />
                      <Text
                        strong
                        style={{
                          fontSize: 22,
                          color: isPass ? "#389e0d" : "#cf1322",
                        }}
                      >
                        {result.prediction_result}
                      </Text>
                    </div>
                  </Flex>
                  <Tag
                    color={isPass ? "success" : "error"}
                    style={{
                      fontSize: 14,
                      padding: "4px 16px",
                      fontWeight: 700,
                      borderRadius: 20,
                    }}
                  >
                    {isPass ? "Đạt" : "Không đạt"}
                  </Tag>
                </Flex>
                <Divider style={{ margin: "0 0 20px" }} />
                <Space direction="vertical" size={16} style={{ width: "100%" }}>
                  <div>
                    <Flex justify="space-between" style={{ marginBottom: 6 }}>
                      <Text strong style={{ color: "#389e0d" }}>
                        Xác suất PASS
                      </Text>
                      <Text strong style={{ color: "#389e0d" }}>
                        {(result.probability_pass * 100).toFixed(1)}%
                      </Text>
                    </Flex>
                    <Progress
                      percent={passPercent}
                      strokeColor={{ from: "#52c41a", to: "#389e0d" }}
                      trailColor="#f0f4fa"
                      status="active"
                      strokeWidth={14}
                      format={() => null}
                    />
                  </div>
                  <div>
                    <Flex justify="space-between" style={{ marginBottom: 6 }}>
                      <Text strong style={{ color: "#cf1322" }}>
                        Xác suất FAIL
                      </Text>
                      <Text strong style={{ color: "#cf1322" }}>
                        {(result.probability_fail * 100).toFixed(1)}%
                      </Text>
                    </Flex>
                    <Progress
                      percent={failPercent}
                      strokeColor={{ from: "#ff7875", to: "#cf1322" }}
                      trailColor="#f0f4fa"
                      status={failPercent > 50 ? "exception" : "normal"}
                      strokeWidth={14}
                      format={() => null}
                    />
                  </div>
                </Space>
                <Divider style={{ margin: "20px 0 16px" }} />
                {risk && (
                  <div
                    style={{
                      background: risk.bg,
                      border: `1px solid ${risk.border}`,
                      borderRadius: 10,
                      padding: "16px 20px",
                    }}
                  >
                    <Flex align="start" gap={12}>
                      <span style={{ fontSize: 20, color: risk.color }}>
                        {risk.icon}
                      </span>
                      <div>
                        <Text
                          strong
                          style={{ color: risk.color, fontSize: 14 }}
                        >
                          {risk.title}
                        </Text>
                        <br />
                        <Text
                          style={{
                            color: "#5a6a85",
                            fontSize: 13,
                            lineHeight: "20px",
                          }}
                        >
                          {risk.description}
                        </Text>
                      </div>
                    </Flex>
                  </div>
                )}
              </Card>
            )}
            {!isPending && !result && !isError && (
              <Card
                style={{
                  textAlign: "center",
                  padding: "80px 0",
                  borderRadius: 16,
                  border: "2px dashed #D7E1F0",
                  boxShadow: "none",
                  background: "#fafbfd",
                }}
              >
                <BarChartOutlined
                  style={{ fontSize: 52, color: "#c8d4e6", marginBottom: 12 }}
                />
                <br />
                <Text type="secondary" style={{ fontSize: 14 }}>
                  Điền thông tin bên trái và nhấn{" "}
                  <Text strong style={{ color: "#004286" }}>
                    Dự Đoán Kết Quả
                  </Text>{" "}
                  để xem phân tích
                </Text>
              </Card>
            )}
          </Col>
        </Row>
      </div>
    </div>
  );
}
