import { useState } from "react";
import {
  Typography,
  Input,
  Button,
  Row,
  Col,
  Card,
  Alert,
  Space,
  Empty,
  Spin,
  Tag,
  Flex,
} from "antd";
import {
  SearchOutlined,
  TrophyOutlined,
  StarFilled,
  TeamOutlined,
  BookOutlined,
} from "@ant-design/icons";
import { useTeacherRecommendation } from "../hooks/useRecommendation";
import type { TeacherItem } from "../api/recommendation.api";

const { Title, Text, Paragraph } = Typography;

const RANK_MEDALS = ["🥇", "🥈", "🥉"];
const RANK_COLORS = ["#e8a838", "#94a3b8", "#c47f17"];

function ScoreBar({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  const percent = Math.round(value * 100);
  return (
    <div style={{ marginBottom: 8 }}>
      <Flex justify="space-between" style={{ marginBottom: 2 }}>
        <Text style={{ fontSize: 12, color: "#5a6a85" }}>{label}</Text>
        <Text strong style={{ fontSize: 12, color }}>
          {value.toFixed(4)}
        </Text>
      </Flex>
      <div
        style={{
          height: 6,
          borderRadius: 3,
          background: "#f0f4fa",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${Math.max(percent, 2)}%`,
            height: "100%",
            borderRadius: 3,
            background: color,
            transition: "width 0.6s ease",
          }}
        />
      </div>
    </div>
  );
}

function TeacherCard({
  teacher,
  rank,
}: {
  teacher: TeacherItem;
  rank: number;
}) {
  const isTop3 = rank <= 3;
  const medal = RANK_MEDALS[rank - 1];
  const rankColor = RANK_COLORS[rank - 1] ?? "#1a3c6e";
  const seenCourses = new Set<string>();
  const courses = teacher.course_name
    .split(/;|,(?![^()]*\))/)
    .map((course) => course.trim())
    .filter((course) => {
      const key = course.normalize("NFC").toLocaleLowerCase("vi");
      if (!key || seenCourses.has(key)) return false;
      seenCourses.add(key);
      return true;
    });

  return (
    <Card
      hoverable
      style={{
        height: "100%",
        borderTop: isTop3 ? `3px solid ${rankColor}` : "3px solid #e0e6f0",
        borderRadius: 12,
        boxShadow: "0 2px 12px rgba(26,60,110,0.08)",
      }}
    >
      <Flex align="center" gap={10} style={{ marginBottom: 16 }}>
        <div
          style={{
            width: 40,
            height: 40,
            borderRadius: "50%",
            background: isTop3
              ? `linear-gradient(135deg, ${rankColor}22, ${rankColor}44)`
              : "#f0f4fa",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: isTop3 ? 20 : 14,
            fontWeight: 700,
            color: rankColor,
          }}
        >
          {isTop3 ? medal : `#${rank}`}
        </div>
        <div style={{ flex: 1 }}>
          <Text strong style={{ fontSize: 15, color: "#1a1a2e" }}>
            {teacher.teacher_name || `Giảng viên #${teacher.teacher_id}`}
          </Text>
          <br />
          <Text style={{ fontSize: 12, color: "#5a6a85" }}>
            Mã GV: T-{String(teacher.teacher_id).padStart(3, "0")}
          </Text>
        </div>
        {isTop3 && (
          <Tag
            color={rankColor}
            style={{ borderRadius: 12, fontWeight: 600, margin: 0 }}
          >
            Top {rank}
          </Tag>
        )}
      </Flex>

      {/* Môn học giảng dạy */}
      {teacher.course_name && (
        <Flex align="center" gap={6} wrap="wrap" style={{ marginBottom: 14 }}>
          <BookOutlined style={{ color: "#004286", fontSize: 13 }} />
          {courses.map((c) => (
            <Tag
              key={c.trim()}
              style={{
                borderRadius: 10,
                fontSize: 11,
                margin: 0,
                maxWidth: "100%",
                whiteSpace: "normal",
                overflowWrap: "anywhere",
                background: "#EEF3FB",
                color: "#004286",
                border: "1px solid #D7E1F0",
              }}
            >
              {c.trim()}
            </Tag>
          ))}
        </Flex>
      )}

      <div
        style={{
          background: "linear-gradient(135deg, #1a3c6e08, #2a529812)",
          borderRadius: 10,
          padding: "12px 14px",
          marginBottom: 14,
        }}
      >
        <Flex align="center" gap={8}>
          <TrophyOutlined style={{ color: "#e8a838", fontSize: 18 }} />
          <div style={{ flex: 1 }}>
            <Text style={{ fontSize: 11, color: "#5a6a85" }}>Điểm Hybrid</Text>
            <br />
            <Text strong style={{ fontSize: 20, color: "#1a3c6e" }}>
              {teacher.hybrid_score.toFixed(4)}
            </Text>
          </div>
        </Flex>
      </div>

      <ScoreBar
        label="Content-Based"
        value={teacher.content_score}
        color="#2a5298"
      />
      <ScoreBar
        label="Collaborative"
        value={teacher.collab_score}
        color="#e8a838"
      />
    </Card>
  );
}

export default function TeacherRecommendationPage() {
  const [query, setQuery] = useState("");
  const { mutate, data, isPending, isError, error } =
    useTeacherRecommendation();

  const handleSearch = () => {
    const trimmed = query.trim();
    if (!trimmed) return;
    mutate({ query_text: trimmed, alpha: 0.6, top_k: 5 });
  };

  const teachers = data?.data?.items ?? [];

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
              content: '""',
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
            <TeamOutlined style={{ fontSize: 26, color: "#F2C94C" }} />
            <Title
              level={2}
              style={{ margin: 0, color: "#fff", fontWeight: 700 }}
            >
              Gợi Ý Giảng Viên Phù Hợp
            </Title>
          </Flex>
          <Paragraph
            style={{ margin: 0, fontSize: 15, color: "rgba(255,255,255,0.8)" }}
          >
            Mô tả nhu cầu học tập của bạn — hệ thống AI sẽ phân tích và gợi ý
            giảng viên phù hợp nhất.
          </Paragraph>
        </div>

        {/* Search Box */}
        <Card
          style={{
            marginBottom: 24,
            borderRadius: 16,
            border: "none",
            boxShadow: "0 2px 12px rgba(26,60,110,0.08)",
          }}
        >
          <Input.Search
            size="large"
            allowClear
            placeholder="VD: Tôi muốn học Machine Learning, giảng viên dạy thực hành qua Google Meet..."
            enterButton={
              <Button
                type="primary"
                icon={<SearchOutlined />}
                loading={isPending}
                style={{
                  borderRadius: "0 10px 10px 0",
                  height: "auto",
                  padding: "0 24px",
                }}
              >
                Tìm Kiếm
              </Button>
            }
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onSearch={handleSearch}
            disabled={isPending}
          />
        </Card>

        {/* Error */}
        {isError && (
          <Alert
            type="error"
            message="Lỗi khi tải dữ liệu"
            description={error?.message}
            showIcon
            closable
            style={{ marginBottom: 20, borderRadius: 10 }}
          />
        )}

        {/* Loading */}
        {isPending && (
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
                Đang phân tích và tìm kiếm giảng viên phù hợp...
              </Text>
            </Space>
          </div>
        )}

        {/* Results */}
        {!isPending && teachers.length > 0 && (
          <div>
            <Flex align="center" gap={8} style={{ marginBottom: 16 }}>
              <StarFilled style={{ color: "#F2C94C" }} />
              <Text style={{ color: "#5a6a85" }}>
                Tìm thấy{" "}
                <Text strong style={{ color: "#004286" }}>
                  {teachers.length}
                </Text>{" "}
                giảng viên phù hợp
              </Text>
            </Flex>
            <Row gutter={[20, 20]}>
              {teachers.map((teacher, index) => (
                <Col xs={24} sm={12} lg={8} xl={8} key={teacher.teacher_id}>
                  <TeacherCard teacher={teacher} rank={index + 1} />
                </Col>
              ))}
            </Row>
          </div>
        )}

        {/* Empty after search */}
        {!isPending && !isError && data && teachers.length === 0 && (
          <Card
            style={{
              textAlign: "center",
              padding: "60px 0",
              borderRadius: 16,
              border: "2px dashed #D7E1F0",
              boxShadow: "none",
              background: "#fafbfd",
            }}
          >
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={
                <Text type="secondary">Không tìm thấy giảng viên phù hợp.</Text>
              }
            />
          </Card>
        )}

        {/* Initial state */}
        {!isPending && !isError && !data && (
          <Card
            style={{
              textAlign: "center",
              padding: "60px 0",
              borderRadius: 16,
              border: "2px dashed #D7E1F0",
              boxShadow: "none",
              background: "#fafbfd",
            }}
          >
            <SearchOutlined
              style={{ fontSize: 48, color: "#c8d4e6", marginBottom: 12 }}
            />
            <br />
            <Text type="secondary" style={{ fontSize: 14 }}>
              Nhập mô tả nhu cầu ở thanh tìm kiếm phía trên để bắt đầu
            </Text>
          </Card>
        )}
      </div>
    </div>
  );
}
