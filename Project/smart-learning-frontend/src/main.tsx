import { StrictMode } from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ConfigProvider } from "antd";
import viVN from "antd/locale/vi_VN";

import App from "./App";
import "antd/dist/reset.css";
import "./styles/app.css";

// Query client dùng chung cho toàn app
const queryClient = new QueryClient();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ConfigProvider
      locale={viVN}
      theme={{
        token: {
          colorPrimary: "#004286",
          colorSuccess: "#5FAE6F",
          colorWarning: "#F2C94C",
          colorError: "#E56B6F",
          colorBgLayout: "#EEF3FB",
          colorBgContainer: "#FFFFFF",
          colorText: "#183A70",
          colorBorderSecondary: "#D7E1F0",
          borderRadius: 16,
        },
      }}
    >
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>
    </ConfigProvider>
  </StrictMode>,
);
