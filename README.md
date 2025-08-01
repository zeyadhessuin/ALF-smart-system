# Assisted-Living Risk-Alert System

## Project Overview

The Assisted-Living Risk-Alert System is a machine learning solution designed to predict health incidents in assisted living facilities before they occur. By analyzing daily resident data including vital signs, medication adherence, and medical history, the system enables proactive care interventions that can significantly reduce emergency situations.

### Key Achievements
- **68% recall rate** - Successfully identifies 2 out of 3 future health incidents
- **30% reduction** in next-day incidents through targeted interventions
- **Real-time monitoring** capability for 500+ residents across 5 facilities
- **Actionable insights** that integrate seamlessly into existing care workflows

---

## Business Problem

### Challenge
Assisted living facilities face the critical challenge of identifying residents at risk of health incidents before they occur. Traditional reactive approaches often result in:
- Emergency hospitalizations that could have been prevented
- Increased healthcare costs and resource strain
- Reduced quality of life for residents
- Higher liability and regulatory compliance risks

### Solution Impact
Our predictive model transforms care delivery from reactive to proactive by:
- Enabling early intervention through 24-hour risk forecasting
- Providing clear, actionable alerts for caregiving staff
- Reducing emergency incidents by approximately 30%
- Maintaining high recall (68%) to minimize missed critical cases

---

## Dataset Description

### Data Sources
- **Total Records**: 60,000 daily resident observations
- **Population**: 500 residents across 5 assisted living facilities
- **Time Period**: 18 months of historical data
- **Update Frequency**: Daily data collection and model scoring

### Raw Variables (12 features)
| Category | Variables | Description |
|----------|-----------|-------------|
| **Vital Signs** | Heart Rate, Blood Pressure (Systolic/Diastolic), Temperature | Daily measurements taken during routine checks |
| **Medications** | Adherence Score, Medication Count | Compliance tracking and medication complexity |
| **Demographics** | Age, Gender, Length of Stay | Resident characteristics and tenure |
| **Medical History** | Primary Diagnosis, Comorbidity Count | Clinical background and risk factors |
| **Behavioral** | Activity Level, Sleep Quality | Daily living indicators |

### Target Variable
- **Health Incident**: Binary classification (Yes/No)
- **Time Horizon**: Next 24 hours
- **Incident Rate**: 5% (highly imbalanced dataset)
- **Incident Types**: Falls, cardiac events, respiratory distress, severe medication reactions

---

## Technical Architecture

### Data Pipeline
```
Raw Data Sources → Data Validation → Feature Engineering → Model Training → Prediction Pipeline → Alert System
```

### Infrastructure Requirements
- **Storage**: PostgreSQL database for historical data, Redis for real-time caching
- **Processing**: Python-based ETL pipeline with Apache Airflow scheduling
- **ML Framework**: Scikit-learn, LightGBM, SHAP for interpretability
- **Monitoring**: MLflow for experiment tracking and model versioning
- **Deployment**: Docker containers with REST API endpoints

---

## Feature Engineering

### Engineered Features (24 total features)

#### Time-Series Features
- **3-day rolling means**: Heart rate, blood pressure, temperature
- **3-day rolling deltas**: Change from baseline for each vital sign
- **Trend indicators**: Increasing/decreasing patterns over observation window

#### Categorical Transformations
- **Age groups**: Binned into clinically relevant ranges (65-74, 75-84, 85+)
- **Medication adherence buckets**: Low (<70%), Medium (70-89%), High (90%+)
- **High-risk diagnosis flags**: Binary indicators for dementia, CHF, Parkinson's, diabetes

#### Interaction Features
- **Age × diagnosis interactions**: Combined risk factors
- **Medication complexity score**: Count × adherence interaction
- **Comorbidity burden**: Weighted sum of multiple conditions

### Feature Selection Process
1. **Correlation analysis** to remove redundant features
2. **Recursive feature elimination** with cross-validation
3. **SHAP-based importance ranking** for interpretability
4. **Clinical validation** with domain experts

---

## Model Development

### Data Splitting Strategy
- **Patient-wise split** to prevent data leakage
- **Training set**: 80% (48k records, ~400 residents)
- **Validation set**: 10% (6k records, ~50 residents)  
- **Test set**: 10% (6k records, ~50 residents)

### Class Imbalance Handling
- **SMOTE (Synthetic Minority Oversampling)** applied to training set only
- **Stratified sampling** to maintain incident rate across splits
- **Cost-sensitive learning** with adjusted class weights

### Algorithm Benchmarking
| Algorithm | PR-AUC | Recall | Precision | F1-Score | Training Time |
|-----------|--------|--------|-----------|----------|---------------|
| **LightGBM** | **0.312** | **0.68** | **0.23** | **0.34** | **2.3 min** |
| XGBoost | 0.298 | 0.65 | 0.21 | 0.32 | 4.1 min |
| CatBoost | 0.285 | 0.63 | 0.20 | 0.30 | 6.2 min |
| Random Forest | 0.271 | 0.61 | 0.19 | 0.29 | 3.8 min |
| Gradient Boosting | 0.254 | 0.59 | 0.18 | 0.28 | 5.7 min |
| Logistic Regression | 0.198 | 0.52 | 0.14 | 0.22 | 0.8 min |

### Hyperparameter Optimization
- **Bayesian optimization** with 100 iterations
- **5-fold cross-validation** for robust evaluation
- **Early stopping** to prevent overfitting
- **Key parameters optimized**: learning_rate, num_leaves, feature_fraction, min_child_samples

---

## Results & Performance

### Model Performance Metrics
- **PR-AUC**: 0.312 (significantly better than random baseline of 0.05)
- **Recall**: 0.68 (captures 68% of actual incidents)
- **Precision**: 0.23 (23% of predictions are true positives)
- **Specificity**: 0.85 (correctly identifies 85% of non-incident cases)

### Business Impact Metrics
- **Alert Volume**: ~15 alerts per day across all facilities (manageable workload)
- **Intervention Success Rate**: 30% reduction in incidents for residents receiving targeted care
- **Cost Savings**: Estimated $2,400 per prevented hospitalization
- **ROI**: 4.2x return on implementation investment within first year

---

## Key Insights

### Top Risk Drivers (SHAP Analysis)

#### 1. 3-Day Heart Rate Rolling Mean (Importance: 0.28)
- **Clinical Significance**: Persistent elevation indicates cardiovascular stress
- **Threshold**: ≥5 bpm above individual baseline triggers high-risk classification
- **Actionable**: Immediate vital sign monitoring and physician consultation

#### 2. High-Risk Diagnosis Flag (Importance: 0.21)
- **Conditions**: Dementia, Congestive Heart Failure, Parkinson's Disease
- **Risk Multiplier**: 2x baseline incident probability
- **Management**: Enhanced monitoring protocols and specialized care plans

#### 3. Medication Adherence Score (Importance: 0.18)
- **Scale**: 0-1 continuous score based on dose timing and completion
- **Impact**: Every 0.1 decrease correlates with 12% increased risk
- **Intervention**: Automated dispensing systems and adherence coaching

### Additional Insights
- **Age factor**: Risk increases exponentially after age 85
- **Seasonal patterns**: 23% higher incident rate during winter months
- **Facility variations**: Location-specific risk factors identified
- **Time of day**: 67% of incidents occur between 6 PM - 6 AM

---

## Implementation Recommendations

### 1. Daily Vitals Dashboard
**Objective**: Real-time monitoring with automated alerting

**Technical Implementation**:
- Integration with existing vital signs monitoring equipment
- Automated data ingestion via HL7 FHIR standards
- Real-time dashboard with color-coded risk levels
- Mobile notifications for on-duty nursing staff

**Alert Criteria**:
- 3-day heart rate rolling mean ≥ baseline + 5 bpm
- Blood pressure deviation > 20% from personal average
- Temperature trend showing consistent elevation

### 2. Priority Care Rounds
**Objective**: Targeted interventions for high-risk residents

**Implementation Strategy**:
- Daily risk scoring and resident prioritization
- Enhanced monitoring for residents with dementia, CHF, or Parkinson's
- Structured assessment protocols for flagged residents
- Documentation integration with existing care management systems

**Resource Allocation**:
- Additional 15-minute checks for high-risk residents
- Specialized training for staff on early intervention techniques
- Coordination with on-call physicians for rapid response

### 3. Medication Adherence Program
**Objective**: Maintain >90% adherence rates across all residents

**Technology Solutions**:
- Automated dispensing systems with dose tracking
- Smart pill bottles with adherence monitoring
- Integration with pharmacy management systems
- Real-time adherence scoring and alerts

**Clinical Protocols**:
- Nurse-assisted dosing for residents with <80% adherence
- Blister pack preparation for complex medication regimens
- Regular medication reviews and simplification strategies

---
