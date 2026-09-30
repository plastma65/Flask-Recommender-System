import { useMutation } from "@tanstack/react-query";
import {
  fetchTeacherRecommendations,
  fetchPassPrediction,
} from "../api/recommendation.api";
import type {
  TeacherRecommendationPayload,
  TeacherRecommendationResponse,
  PassPredictionPayload,
  PassPredictionResponse,
} from "../api/recommendation.api";

export const useTeacherRecommendation = () =>
  useMutation<
    TeacherRecommendationResponse,
    Error,
    TeacherRecommendationPayload
  >({
    mutationFn: fetchTeacherRecommendations,
  });

export const usePassPrediction = () =>
  useMutation<PassPredictionResponse, Error, PassPredictionPayload>({
    mutationFn: fetchPassPrediction,
  });
