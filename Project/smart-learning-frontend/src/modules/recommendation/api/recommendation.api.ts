import axios from "axios";

const apiClient = axios.create({
  baseURL: "/api/v1",
  headers: { "Content-Type": "application/json" },
  timeout: 30000,
});

apiClient.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message =
      error.response?.data?.detail ??
      error.response?.data?.message ??
      error.message ??
      "Đã có lỗi xảy ra. Vui lòng thử lại.";
    return Promise.reject(new Error(message));
  },
);

export type TeacherRecommendationPayload = {
  student_id?: string | number;
  query_text: string;
  alpha: number;
  top_k: number;
};

export type TeacherItem = {
  teacher_id: number;
  teacher_name: string;
  content_score: number;
  collab_score: number;
  hybrid_score: number;
  course_name: string;
};

export type TeacherRecommendationResponse = {
  success: boolean;
  data: { items: TeacherItem[] };
};

export type PassPredictionPayload = {
  gpa_bucket: string;
  study_hours_bucket: string;
  failed_subjects_count: string;
  attendance_bucket: string;
};

export type PassPredictionResult = {
  probability_pass: number;
  probability_fail: number;
  prediction_result: string;
  risk_level: string;
};

export type PassPredictionResponse = {
  success: boolean;
  data: PassPredictionResult;
};

export const fetchTeacherRecommendations = (
  payload: TeacherRecommendationPayload,
) =>
  apiClient.post<never, TeacherRecommendationResponse>(
    "/recommendations/teachers",
    payload,
  );

export const fetchPassPrediction = (payload: PassPredictionPayload) =>
  apiClient.post<never, PassPredictionResponse>(
    "/predictions/pass-fail",
    payload,
  );
