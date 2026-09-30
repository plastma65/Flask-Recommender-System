import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import MainLayout from "../layouts/MainLayout";
import TeacherRecommendationPage from "../modules/recommendation/pages/TeacherRecommendationPage";
import PassPredictionPage from "../modules/recommendation/pages/PassPredictionPage";

// Toàn bộ route của đề tài Recommendation System
export default function AppRoutes() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<MainLayout />}>
          <Route
            path="/"
            element={<Navigate to="/recommendations/teachers" replace />}
          />
          <Route
            path="/recommendations/teachers"
            element={<TeacherRecommendationPage />}
          />
          <Route
            path="/recommendations/prediction"
            element={<PassPredictionPage />}
          />
        </Route>
        <Route
          path="*"
          element={<Navigate to="/recommendations/teachers" replace />}
        />
      </Routes>
    </BrowserRouter>
  );
}
