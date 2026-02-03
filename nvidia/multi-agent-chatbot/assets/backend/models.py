#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from pydantic import BaseModel
from typing import Optional, List, Dict

class ChatConfig(BaseModel):
    sources: List[str]
    models :  List[str]
    selected_model: Optional[str] = None
    selected_sources: Optional[List[str]] = None
    current_chat_id: Optional[str] = None

class ChatIdRequest(BaseModel):
    chat_id: str

class ChatRenameRequest(BaseModel):
    chat_id: str
    new_name: str

class SelectedModelRequest(BaseModel):
    model: str

class ImageDescriptionRequest(BaseModel):
    image_id: str
    description: str

class BatchAnalysisRequest(BaseModel):
    image_ids: List[str]
    analysis_prompt: str
    report_format: Optional[str] = "markdown"
    organization: Optional[str] = None
    descriptions_map: Optional[Dict[str, str]] = None  # {filename: description} mapping

class CronJobRequest(BaseModel):
    name: str
    schedule: str  # Cron expression: "minute hour day month day_of_week"
    external_api_get_url: str
    external_api_post_url: str
    external_api_key: str
    backend_api_url: Optional[str] = "http://localhost:8000"
    organization_id: str
    enabled: Optional[bool] = True

class CronJobUpdateRequest(BaseModel):
    enabled: Optional[bool] = None
    schedule: Optional[str] = None
    external_api_get_url: Optional[str] = None
    external_api_post_url: Optional[str] = None
    external_api_key: Optional[str] = None
    organization_id: Optional[str] = None


# Breakdown Prediction Models

class SensorReading(BaseModel):
    """Individual sensor reading."""
    name: str
    value: float
    unit: Optional[str] = None
    timestamp: Optional[str] = None
    status: Optional[str] = "normal"  # normal, warning, critical
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None


class AssetInfo(BaseModel):
    """Asset information for breakdown prediction."""
    asset_id: str
    asset_name: str
    asset_type: Optional[str] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None
    installation_date: Optional[str] = None
    last_maintenance_date: Optional[str] = None
    operating_hours: Optional[float] = None
    criticality: Optional[str] = None  # critical, high, medium, low
    location: Optional[str] = None
    department: Optional[str] = None
    additional_info: Optional[Dict] = None


class BreakdownPredictionRequest(BaseModel):
    """Request body for single asset breakdown prediction."""
    asset_info: AssetInfo
    sensor_data: List[SensorReading]
    collection_name: Optional[str] = None  # Vector DB collection for RAG


class BreakdownPredictionDirectRequest(BaseModel):
    """Request body for breakdown prediction with optional direct data input.

    This model allows N8N and other integrations to provide historical data directly
    instead of retrieving from vector DB. If a field is not provided, the system
    will fall back to vector DB retrieval.

    Accepts either structured JSON objects (List[Dict]) or pre-formatted text strings.
    """
    asset_info: Dict  # Full asset info object from external system
    sensor_data: Optional[List[Dict]] = None  # Live sensor readings
    # Optional direct input - accepts structured JSON or pre-formatted text
    work_orders: Optional[List[Dict] | str] = None  # Work orders as JSON array or text
    inspections: Optional[List[Dict] | str] = None  # Inspections as JSON array or text
    service_schedules: Optional[List[Dict] | str] = None  # Service schedules as JSON array or text
    maintenance_requests: Optional[List[Dict] | str] = None  # Maintenance requests as JSON array or text
    knowledge_base: Optional[str] = None  # Pre-formatted knowledge base text
    # Vector DB collection for fallback retrieval
    collection_name: Optional[str] = None


class BatchBreakdownPredictionRequest(BaseModel):
    """Request body for batch breakdown prediction."""
    assets: List[Dict]  # Each dict contains asset_info and sensor_data
    collection_name: Optional[str] = None


class BreakdownPrediction(BaseModel):
    """Individual breakdown prediction."""
    failure_mode: str
    probability: float
    estimated_timeframe: str
    risk_score: float
    risk_level: str


class ContributingFactor(BaseModel):
    """Factor contributing to potential breakdown."""
    factor: str
    category: str
    severity: str
    evidence: str
    trend: str


class DowntimeEstimate(BaseModel):
    """Estimated downtime range."""
    minimum: float
    maximum: float
    most_likely: float


class CostEstimate(BaseModel):
    """Estimated cost range."""
    minimum: float
    maximum: float
    most_likely: float


class DowntimeImpactAnalysis(BaseModel):
    """Downtime impact analysis."""
    estimated_downtime_hours: DowntimeEstimate
    production_impact: str
    safety_implications: str
    environmental_risk: str


class CostAnalysis(BaseModel):
    """Cost analysis for potential breakdown."""
    estimated_repair_cost: CostEstimate
    production_loss_cost_per_hour: float
    total_potential_cost: CostEstimate
    preventive_action_cost_estimate: float
    cost_avoidance_potential: float


class PreventionRecommendation(BaseModel):
    """Recommended action to prevent breakdown."""
    priority: int
    action: str
    category: str
    estimated_time_to_complete: str
    required_resources: List[str]
    expected_risk_reduction: str
    deadline: str


class MonitoringRecommendation(BaseModel):
    """Monitoring recommendation for a parameter."""
    parameter: str
    current_value: str
    threshold_warning: str
    threshold_critical: str
    monitoring_frequency: str
    escalation_procedure: str


class DataQualityNotes(BaseModel):
    """Notes on data quality and gaps."""
    data_completeness: str
    data_gaps_identified: List[str]
    recommendations_for_better_prediction: List[str]


class PredictionSummary(BaseModel):
    """Summary of breakdown prediction."""
    asset_id: str
    asset_name: str
    analysis_timestamp: str
    overall_risk_level: str
    breakdown_probability_30_days: float
    confidence_level: str
    recommended_action_urgency: str


class WorkOrderRecord(BaseModel):
    """Individual work order record in thought process."""
    work_order_id: str
    date: Optional[str] = None
    issue_type: str
    description: str
    relevance: str
    failure_pattern_indicator: str


class WorkOrdersAnalysis(BaseModel):
    """Work orders analysis in thought process."""
    records_reviewed: int
    relevant_records: List[WorkOrderRecord]
    patterns_identified: List[str]
    mtbf_calculation: str
    recurring_issues: List[str]
    key_insights: str


class InspectionRecord(BaseModel):
    """Individual inspection record in thought process."""
    inspection_id: str
    date: Optional[str] = None
    inspection_type: str
    findings: str
    condition_rating: Optional[str] = None
    relevance: str
    degradation_indicator: str


class InspectionsAnalysis(BaseModel):
    """Inspections analysis in thought process."""
    records_reviewed: int
    relevant_records: List[InspectionRecord]
    condition_trend: str
    critical_findings: List[str]
    watch_items: List[str]
    key_insights: str


class ServiceScheduleRecord(BaseModel):
    """Individual service schedule record in thought process."""
    schedule_id: str
    maintenance_type: str
    frequency: str
    last_performed: Optional[str] = None
    next_due: Optional[str] = None
    compliance_status: str
    relevance: str


class ServiceSchedulesAnalysis(BaseModel):
    """Service schedules analysis in thought process."""
    records_reviewed: int
    relevant_records: List[ServiceScheduleRecord]
    overdue_maintenance: List[str]
    upcoming_critical_maintenance: List[str]
    compliance_rate: str
    maintenance_gaps: List[str]
    key_insights: str


class SensorReadingAnalysis(BaseModel):
    """Individual sensor reading analysis in thought process."""
    sensor_name: str
    parameter: str
    current_value: str
    normal_range: str
    status: str
    trend: str
    why_monitored: str
    anomaly_detected: bool
    anomaly_description: Optional[str] = None
    failure_correlation: str


class SensorCorrelation(BaseModel):
    """Sensor correlation in thought process."""
    sensors: List[str]
    correlation_type: str
    significance: str


class SensorDataAnalysis(BaseModel):
    """Sensor data analysis in thought process."""
    sensors_monitored: int
    sensor_readings: List[SensorReadingAnalysis]
    critical_alerts: List[str]
    anomaly_summary: str
    sensor_correlations: List[SensorCorrelation]
    key_insights: str


class CrossDataCorrelation(BaseModel):
    """Cross-data correlation in thought process."""
    data_sources: List[str]
    correlation: str
    implication: str
    confidence: str


class RiskFactor(BaseModel):
    """Risk factor with weight and justification."""
    factor: str
    weight: str
    justification: str


class RiskCalculationReasoning(BaseModel):
    """Risk calculation reasoning in thought process."""
    probability_factors: List[RiskFactor]
    severity_factors: List[RiskFactor]
    detection_factors: List[RiskFactor]
    final_risk_score_calculation: str


class ConfidenceFactors(BaseModel):
    """Confidence factors assessment."""
    data_quantity: str
    data_recency: str
    data_consistency: str
    pattern_clarity: str


class ConfidenceAssessment(BaseModel):
    """Confidence assessment in thought process."""
    overall_confidence: str
    confidence_factors: ConfidenceFactors
    limitations: List[str]
    assumptions_made: List[str]


class DecisionLogicStep(BaseModel):
    """Individual step in decision logic chain."""
    step: int
    observation: str
    inference: str
    confidence: str


class AlternativeScenario(BaseModel):
    """Alternative scenario considered."""
    scenario: str
    probability: str
    why_not_primary: str


class ThoughtProcess(BaseModel):
    """Comprehensive thought process for breakdown prediction explainability."""
    executive_summary: str
    work_orders_analysis: WorkOrdersAnalysis
    inspections_analysis: InspectionsAnalysis
    service_schedules_analysis: ServiceSchedulesAnalysis
    sensor_data_analysis: SensorDataAnalysis
    cross_data_correlations: List[CrossDataCorrelation]
    risk_calculation_reasoning: RiskCalculationReasoning
    confidence_assessment: ConfidenceAssessment
    decision_logic_chain: List[DecisionLogicStep]
    alternative_scenarios_considered: List[AlternativeScenario]


class BreakdownPredictionResponse(BaseModel):
    """Complete breakdown prediction response."""
    prediction_id: str
    asset_id: str
    asset_name: str
    generated_at: str
    prediction: Dict  # Contains the full prediction JSON including thought_process
    data_sources: Dict[str, bool]
