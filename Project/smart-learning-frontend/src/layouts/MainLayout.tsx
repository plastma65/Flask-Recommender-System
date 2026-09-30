import { useState } from "react";
import { Layout, Menu, Typography } from "antd";
import { TrophyOutlined, ExperimentOutlined } from "@ant-design/icons";
import { Outlet, useLocation, useNavigate } from "react-router-dom";

import stuLogo from "../assets/stu-logo.png";
import stuTopbar from "../assets/STU-topbar.png";

const { Sider, Header, Content } = Layout;
const { Title, Text } = Typography;

const SIDER_WIDTH = 260;
const SIDER_COLLAPSED_WIDTH = 80;
const LOGO_SIZE = 44;
// Canh logo sao cho tâm logo trùng tâm rail khi thu gọn
// => logo đứng yên tuyệt đối trong lúc sidebar co/giãn.
const LOGO_INSET = (SIDER_COLLAPSED_WIDTH - LOGO_SIZE) / 2;

const MENU_ITEMS = [
  {
    key: "/recommendations/teachers",
    icon: <TrophyOutlined />,
    label: "Gợi ý giảng viên",
  },
  {
    key: "/recommendations/prediction",
    icon: <ExperimentOutlined />,
    label: "Dự đoán Đậu/Rớt",
  },
];

// Layout tối giản riêng cho đề tài Recommendation System
export default function MainLayout() {
  const [collapsed, setCollapsed] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  const activeKey =
    MENU_ITEMS.find((item) => location.pathname.startsWith(item.key))?.key ??
    MENU_ITEMS[0].key;

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Sider
        collapsible
        collapsed={collapsed}
        onCollapse={setCollapsed}
        width={SIDER_WIDTH}
        collapsedWidth={SIDER_COLLAPSED_WIDTH}
        style={{ background: "#004286" }}
      >
        {/* Khối logo: logo giữ nguyên vị trí, chỉ phần chữ mờ dần */}
        <div
          style={{
            height: 84,
            display: "flex",
            alignItems: "center",
            paddingInlineStart: LOGO_INSET,
            overflow: "hidden",
          }}
        >
          <img
            src={stuLogo}
            alt="Logo Trường Đại học Công nghệ Sài Gòn"
            style={{
              width: LOGO_SIZE,
              height: LOGO_SIZE,
              flexShrink: 0,
              borderRadius: 12,
              background: "#fff",
              objectFit: "contain",
              padding: 3,
            }}
          />
          <div
            style={{
              marginInlineStart: 12,
              lineHeight: 1.25,
              whiteSpace: "nowrap",
              overflow: "hidden",
              opacity: collapsed ? 0 : 1,
              transform: collapsed ? "translateX(-8px)" : "none",
              pointerEvents: collapsed ? "none" : "auto",
              transition: "opacity .18s ease, transform .18s ease",
            }}
          >
            <Text
              strong
              style={{ color: "#fff", fontSize: 15, display: "block" }}
            >
              Smart Learning
            </Text>
            <Text style={{ color: "rgba(255,255,255,0.7)", fontSize: 11.5 }}>
              Đại học Công nghệ Sài Gòn
            </Text>
          </div>
        </div>

        <Menu
          theme="dark"
          mode="inline"
          inlineCollapsed={collapsed}
          selectedKeys={[activeKey]}
          items={MENU_ITEMS}
          onClick={({ key }) => navigate(key)}
          style={{ background: "transparent", borderInlineEnd: "none" }}
        />
      </Sider>

      <Layout>
        <Header
          style={{
            background: "#fff",
            borderBottom: "1px solid #D7E1F0",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 24,
            padding: "0 28px",
            height: 72,
          }}
        >
          <Title level={4} style={{ margin: 0, color: "#183A70" }}>
            Hệ thống Gợi ý Học tập
          </Title>
          {/* Logo ngang của trường */}
          <img
            src={stuTopbar}
            alt="Trường Đại học Công nghệ Sài Gòn"
            style={{ height: 42, objectFit: "contain" }}
          />
        </Header>

        <Content style={{ padding: 28 }}>
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  );
}
