import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries at the top level
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools
import os
import json

# Set page configuration
st.set_page_config(
    page_title="Drug Suitability Analysis Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .patient-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ffcccc;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 5px solid #ff0000;
    }
    .alert-medium {
        background-color: #fff4cc;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 5px solid #ffcc00;
    }
    .alert-low {
        background-color: #ccffcc;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 5px solid #00cc00;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    .form-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .knowledge-graph {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .risk-high {
        color: #ff0000;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff9900;
        font-weight: bold;
    }
    .risk-low {
        color: #00cc00;
        font-weight: bold;
    }
    .delete-btn {
        background-color: #ff4444;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
    }
    .delete-btn:hover {
        background-color: #cc0000;
    }
    .edit-btn {
        background-color: #ffaa00;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
    }
    .edit-btn:hover {
        background-color: #cc8800;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# DATA STORAGE MANAGEMENT
# ===============================

class DataStorageManager:
    def __init__(self):
        self.data_dir = "patient_data"
        self.patient_file = os.path.join(self.data_dir, "patients.json")
        self.prescription_file = os.path.join(self.data_dir, "prescriptions.json")
        self._ensure_data_directory()
    
    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def save_patients(self, patient_df):
        """Save patient data to JSON file"""
        try:
            # Convert DataFrame to dictionary
            patient_data = patient_df.to_dict('records')
            with open(self.patient_file, 'w') as f:
                json.dump(patient_data, f, indent=2, default=str)
            return True
        except Exception as e:
            st.error(f"Error saving patient data: {e}")
            return False
    
    def save_prescriptions(self, prescription_df):
        """Save prescription data to JSON file"""
        try:
            prescription_data = prescription_df.to_dict('records')
            with open(self.prescription_file, 'w') as f:
                json.dump(prescription_data, f, indent=2, default=str)
            return True
        except Exception as e:
            st.error(f"Error saving prescription data: {e}")
            return False
    
    def load_patients(self):
        """Load patient data from JSON file"""
        try:
            if os.path.exists(self.patient_file):
                with open(self.patient_file, 'r') as f:
                    patient_data = json.load(f)
                return pd.DataFrame(patient_data)
            return None
        except Exception as e:
            st.error(f"Error loading patient data: {e}")
            return None
    
    def load_prescriptions(self):
        """Load prescription data from JSON file"""
        try:
            if os.path.exists(self.prescription_file):
                with open(self.prescription_file, 'r') as f:
                    prescription_data = json.load(f)
                return pd.DataFrame(prescription_data)
            return None
        except Exception as e:
            st.error(f"Error loading prescription data: {e}")
            return None
    
    def backup_data(self):
        """Create backup of current data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(self.data_dir, "backups")
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            
            if os.path.exists(self.patient_file):
                backup_file = os.path.join(backup_dir, f"patients_backup_{timestamp}.json")
                with open(self.patient_file, 'r') as source, open(backup_file, 'w') as target:
                    target.write(source.read())
            
            if os.path.exists(self.prescription_file):
                backup_file = os.path.join(backup_dir, f"prescriptions_backup_{timestamp}.json")
                with open(self.prescription_file, 'r') as source, open(backup_file, 'w') as target:
                    target.write(source.read())
            
            return True
        except Exception as e:
            st.error(f"Error creating backup: {e}")
            return False

# ===============================
# KNOWLEDGE-BASED AI AGENT
# ===============================

class KnowledgeBasedAIAgent:
    def __init__(self, patient_df, prescription_df):
        self.patient_df = patient_df
        self.prescription_df = prescription_df
        self.semantic_net = nx.Graph()
        self.knowledge_base = self._build_medical_knowledge_base()
        
    def _build_medical_knowledge_base(self):
        """Build comprehensive medical knowledge base"""
        return {
            'disease_drug_relationships': {
                'Hypertension': ['Lisinopril', 'Metoprolol', 'Amlodipine', 'Losartan', 'Hydrochlorothiazide'],
                'Diabetes': ['Metformin', 'Insulin', 'Glipizide', 'Pioglitazone', 'Sitagliptin'],
                'Asthma': ['Albuterol', 'Salmeterol', 'Fluticasone', 'Montelukast', 'Theophylline'],
                'Heart Disease': ['Atorvastatin', 'Simvastatin', 'Warfarin', 'Clopidogrel', 'Aspirin'],
                'Depression': ['Sertraline', 'Fluoxetine', 'Citalopram', 'Venlafaxine', 'Bupropion'],
                'Hyperlipidemia': ['Atorvastatin', 'Simvastatin', 'Rosuvastatin', 'Fenofibrate', 'Ezetimibe'],
                'Thyroid Disorder': ['Levothyroxine', 'Liothyronine', 'Methimazole', 'Propylthiouracil'],
                'Chronic Pain': ['Gabapentin', 'Pregabalin', 'Duloxetine', 'Amitriptyline', 'Tramadol']
            },
            
            'drug_contraindications': {
                'Warfarin': ['Liver Disease', 'Bleeding Disorders', 'Pregnancy', 'Recent Surgery'],
                'Metformin': ['Severe Kidney Disease', 'Liver Failure', 'Metabolic Acidosis'],
                'Lisinopril': ['Pregnancy', 'Angioedema History', 'Bilateral Renal Artery Stenosis'],
                'Atorvastatin': ['Active Liver Disease', 'Pregnancy', 'Breastfeeding'],
                'Metoprolol': ['Asthma', 'Heart Block', 'Cardiogenic Shock', 'Severe Bradycardia'],
                'Gabapentin': ['Severe Respiratory Depression', 'Acute Pancreatitis'],
                'Sertraline': ['Monoamine Oxidase Inhibitors', 'Liver Impairment', 'Seizure Disorders']
            },
            
            'risk_factors': {
                'age_risk': {'>70': 0.3, '>60': 0.2, '>50': 0.1, '>40': 0.05},
                'comorbidity_risk': {'>8': 0.4, '>6': 0.3, '>4': 0.2, '>2': 0.1},
                'polypharmacy_risk': {'>10': 0.5, '>7': 0.3, '>5': 0.2, '>3': 0.1},
                'organ_dysfunction': {'severe': 0.4, 'moderate': 0.2, 'mild': 0.1}
            },
            
            'clinical_pathways': {
                'high_risk_elderly': ['Comprehensive geriatric assessment', 'Medication review', 'Fall risk assessment'],
                'diabetes_management': ['HbA1c monitoring', 'Foot examination', 'Retinal screening', 'Renal function tests'],
                'hypertension_followup': ['BP monitoring', 'Renal function', 'Electrolytes', 'ECG if symptomatic'],
                'polypharmacy_review': ['Deprescribing assessment', 'Drug interaction check', 'Adherence evaluation']
            },
            
            'drug_monitoring_requirements': {
                'Warfarin': ['INR weekly', 'CBC monthly', 'Watch for bleeding signs'],
                'Diuretics': ['Electrolytes monthly', 'Renal function', 'Blood pressure'],
                'ACE Inhibitors': ['Renal function', 'Potassium levels', 'Blood pressure'],
                'Statins': ['Liver enzymes', 'CPK levels', 'Blood glucose'],
                'Metformin': ['Renal function', 'Vitamin B12 levels', 'Lactic acid if symptomatic']
            }
        }
    
    def build_patient_semantic_net(self, patient_id):
        """Step 1: Build Patient-Centric Semantic Network"""
        patient_data = self.patient_df[self.patient_df['patient_id'] == patient_id].iloc[0]
        prescriptions = self.prescription_df[self.prescription_df['patient_id'] == patient_id]
        
        # Clear previous graph
        self.semantic_net = nx.Graph()
        
        # Add patient as central node
        self.semantic_net.add_node(patient_id, type='patient', 
                                 age=patient_data['age'], 
                                 gender=patient_data['gender'],
                                 comorbidity_score=patient_data.get('comorbidity_score', 0))
        
        # Add medical conditions
        conditions = patient_data['medical_conditions']
        for condition in conditions:
            condition_node = f"COND_{condition}"
            self.semantic_net.add_node(condition_node, type='condition', name=condition)
            self.semantic_net.add_edge(patient_id, condition_node, relationship='has_condition')
            
            # Connect conditions to appropriate drugs from knowledge base
            if condition in self.knowledge_base['disease_drug_relationships']:
                for drug in self.knowledge_base['disease_drug_relationships'][condition]:
                    drug_node = f"DRUG_{drug}"
                    self.semantic_net.add_node(drug_node, type='drug', name=drug)
                    self.semantic_net.add_edge(condition_node, drug_node, relationship='treated_with')
        
        # Add current medications
        for _, prescription in prescriptions.iterrows():
            drug_name = prescription['drug_name']
            drug_node = f"DRUG_{drug_name}"
            self.semantic_net.add_node(drug_node, type='drug', name=drug_name, 
                                     dosage=prescription['dosage'],
                                     current=True)
            self.semantic_net.add_edge(patient_id, drug_node, relationship='taking_medication')
        
        # Add allergies
        allergies = patient_data['allergies']
        for allergy in allergies:
            if allergy != 'None':
                allergy_node = f"ALLERGY_{allergy}"
                self.semantic_net.add_node(allergy_node, type='allergy', name=allergy)
                self.semantic_net.add_edge(patient_id, allergy_node, relationship='has_allergy')
        
        # Add organ function nodes
        kidney_node = f"ORGAN_Kidney_{patient_data.get('kidney_function', 'normal')}"
        self.semantic_net.add_node(kidney_node, type='organ', name='Kidney', 
                                 function=patient_data.get('kidney_function', 'normal'))
        self.semantic_net.add_edge(patient_id, kidney_node, relationship='organ_function')
        
        liver_node = f"ORGAN_Liver_{patient_data.get('liver_function', 'normal')}"
        self.semantic_net.add_node(liver_node, type='organ', name='Liver', 
                                 function=patient_data.get('liver_function', 'normal'))
        self.semantic_net.add_edge(patient_id, liver_node, relationship='organ_function')
        
        return self.semantic_net
    
    def impart_knowledge_and_predict(self, patient_id):
        """Step 2: Impart Knowledge and Generate Predictions"""
        patient_data = self.patient_df[self.patient_df['patient_id'] == patient_id].iloc[0]
        prescriptions = self.prescription_df[self.prescription_df['patient_id'] == patient_id]
        
        # Build semantic network first
        self.build_patient_semantic_net(patient_id)
        
        predictions = {
            'patient_id': patient_id,
            'risk_assessment': self._assess_patient_risk(patient_data, prescriptions),
            'drug_suitability': self._assess_drug_suitability(patient_data, prescriptions),
            'care_gaps': self._identify_care_gaps(patient_data, prescriptions),
            'knowledge_insights': self._generate_knowledge_insights(patient_data, prescriptions)
        }
        
        return predictions
    
    def _assess_patient_risk(self, patient_data, prescriptions):
        """Assess comprehensive patient risk using medical knowledge"""
        risk_factors = []
        risk_score = 0.0
        
        # Age-based risk
        age = patient_data['age']
        if age > 70:
            risk_score += 0.3
            risk_factors.append(f"Advanced age ({age} years)")
        elif age > 60:
            risk_score += 0.2
            risk_factors.append(f"Elderly patient ({age} years)")
        elif age > 50:
            risk_score += 0.1
            risk_factors.append(f"Middle-aged ({age} years)")
        
        # Comorbidity risk
        comorbidity_score = patient_data.get('comorbidity_score', 0)
        if comorbidity_score > 8:
            risk_score += 0.4
            risk_factors.append(f"High comorbidity burden (score: {comorbidity_score})")
        elif comorbidity_score > 6:
            risk_score += 0.3
            risk_factors.append(f"Moderate-high comorbidities (score: {comorbidity_score})")
        elif comorbidity_score > 4:
            risk_score += 0.2
            risk_factors.append(f"Moderate comorbidities (score: {comorbidity_score})")
        
        # Polypharmacy risk
        num_medications = len(prescriptions)
        if num_medications > 10:
            risk_score += 0.5
            risk_factors.append(f"Severe polypharmacy ({num_medications} medications)")
        elif num_medications > 7:
            risk_score += 0.3
            risk_factors.append(f"Moderate polypharmacy ({num_medications} medications)")
        elif num_medications > 5:
            risk_score += 0.2
            risk_factors.append(f"Multiple medications ({num_medications} medications)")
        
        # Organ dysfunction risk
        kidney_function = patient_data.get('kidney_function', 'normal')
        if kidney_function == 'severe':
            risk_score += 0.4
            risk_factors.append("Severe kidney dysfunction")
        elif kidney_function == 'moderate':
            risk_score += 0.2
            risk_factors.append("Moderate kidney dysfunction")
        
        liver_function = patient_data.get('liver_function', 'normal')
        if liver_function == 'severe':
            risk_score += 0.4
            risk_factors.append("Severe liver dysfunction")
        elif liver_function == 'moderate':
            risk_score += 0.2
            risk_factors.append("Moderate liver dysfunction")
        
        # Previous ADR history
        if patient_data.get('previous_adr_history', 0) == 1:
            risk_score += 0.3
            risk_factors.append("Previous adverse drug reaction history")
        
        # Normalize risk score
        risk_score = min(risk_score, 1.0)
        
        # Determine risk level
        if risk_score > 0.7:
            risk_level = "High"
        elif risk_score > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'num_risk_factors': len(risk_factors)
        }
    
    def _assess_drug_suitability(self, patient_data, prescriptions):
        """Assess suitability of current medications using medical knowledge"""
        suitability_issues = []
        recommendations = []
        
        conditions = patient_data['medical_conditions']
        allergies = patient_data['allergies']
        
        for _, prescription in prescriptions.iterrows():
            drug_name = prescription['drug_name']
            issues = []
            
            # Check for contraindications
            if drug_name in self.knowledge_base['drug_contraindications']:
                contraindications = self.knowledge_base['drug_contraindications'][drug_name]
                for condition in conditions:
                    if condition in contraindications:
                        issues.append(f"Contraindicated in {condition}")
            
            # Check for appropriate indications
            appropriate_indication = False
            for condition, drugs in self.knowledge_base['disease_drug_relationships'].items():
                if condition in conditions and drug_name in drugs:
                    appropriate_indication = True
                    break
            
            if not appropriate_indication:
                issues.append("No clear indication for current conditions")
            
            # Check allergy interactions
            for allergy in allergies:
                if allergy != 'None' and allergy.lower() in drug_name.lower():
                    issues.append(f"Potential allergy concern: {allergy}")
            
            # Check organ function considerations
            kidney_function = patient_data.get('kidney_function', 'normal')
            liver_function = patient_data.get('liver_function', 'normal')
            
            renal_risk_drugs = ['Metformin', 'Gabapentin', 'Lisnopril', 'Digoxin']
            hepatic_risk_drugs = ['Atorvastatin', 'Simvastatin', 'Warfarin', 'Paracetamol']
            
            if drug_name in renal_risk_drugs and kidney_function in ['moderate', 'severe']:
                issues.append(f"Requires dose adjustment for {kidney_function} kidney function")
            
            if drug_name in hepatic_risk_drugs and liver_function in ['moderate', 'severe']:
                issues.append(f"Requires monitoring for {liver_function} liver function")
            
            if issues:
                suitability_issues.append({
                    'drug_name': drug_name,
                    'issues': issues,
                    'severity': 'High' if any('Contraindicated' in issue for issue in issues) else 'Medium'
                })
            else:
                recommendations.append(f"{drug_name} appears appropriate")
        
        return {
            'suitability_issues': suitability_issues,
            'recommendations': recommendations,
            'total_issues': len(suitability_issues)
        }
    
    def _identify_care_gaps(self, patient_data, prescriptions):
        """Identify gaps in care using medical knowledge"""
        care_gaps = []
        conditions = patient_data['medical_conditions']
        
        # Check for untreated conditions
        for condition in conditions:
            if condition in self.knowledge_base['disease_drug_relationships']:
                appropriate_drugs = self.knowledge_base['disease_drug_relationships'][condition]
                current_drugs = prescriptions['drug_name'].tolist()
                
                # Check if any appropriate drug is being used
                appropriate_treatment = any(drug in current_drugs for drug in appropriate_drugs)
                
                if not appropriate_treatment:
                    care_gaps.append({
                        'type': 'untreated_condition',
                        'condition': condition,
                        'message': f"No appropriate treatment for {condition}",
                        'suggested_treatments': appropriate_drugs[:3]  # Top 3 suggestions
                    })
        
        # Check monitoring requirements
        for _, prescription in prescriptions.iterrows():
            drug_name = prescription['drug_name']
            if drug_name in self.knowledge_base['drug_monitoring_requirements']:
                monitoring_needs = self.knowledge_base['drug_monitoring_requirements'][drug_name]
                care_gaps.append({
                    'type': 'monitoring_required',
                    'drug': drug_name,
                    'message': f"Monitoring required for {drug_name}",
                    'monitoring_tests': monitoring_needs
                })
        
        # Age-specific care gaps
        age = patient_data['age']
        if age > 65:
            care_gaps.append({
                'type': 'geriatric_assessment',
                'message': "Comprehensive geriatric assessment recommended",
                'suggested_actions': ['Fall risk assessment', 'Cognitive screening', 'Functional status evaluation']
            })
        
        if age > 50 and 'Diabetes' not in conditions and 'Hypertension' not in conditions:
            care_gaps.append({
                'type': 'preventive_screening',
                'message': "Consider screening for diabetes and hypertension",
                'suggested_actions': ['Fasting blood glucose', 'HbA1c', 'Blood pressure monitoring']
            })
        
        return care_gaps
    
    def _generate_knowledge_insights(self, patient_data, prescriptions):
        """Generate insights based on medical knowledge"""
        insights = []
        conditions = patient_data['medical_conditions']
        current_drugs = prescriptions['drug_name'].tolist()
        
        # Drug-drug interaction insights
        drug_combinations = list(itertools.combinations(current_drugs, 2))
        for drug1, drug2 in drug_combinations:
            # Known problematic combinations (simplified)
            problematic_pairs = {
                ('Warfarin', 'Aspirin'): "Increased bleeding risk",
                ('Warfarin', 'Ibuprofen'): "Increased bleeding risk",
                ('Lisinopril', 'Spironolactone'): "Hyperkalemia risk",
                ('Metformin', 'Contrast_media'): "Lactic acidosis risk"
            }
            
            pair_key = tuple(sorted([drug1, drug2]))
            if pair_key in problematic_pairs:
                insights.append({
                    'type': 'drug_interaction',
                    'message': f"Potential interaction: {drug1} + {drug2}",
                    'detail': problematic_pairs[pair_key],
                    'severity': 'High'
                })
        
        # Condition-based insights
        if 'Diabetes' in conditions and 'Hypertension' in conditions:
            insights.append({
                'type': 'condition_management',
                'message': "Diabetes and hypertension comorbidity detected",
                'detail': "Tight blood pressure control (<130/80) recommended for renal protection",
                'severity': 'Medium'
            })
        
        if len(current_drugs) > 5:
            insights.append({
                'type': 'polypharmacy',
                'message': f"Polypharmacy detected ({len(current_drugs)} medications)",
                'detail': "Consider deprescribing review and medication reconciliation",
                'severity': 'Medium'
            })
        
        # Age-based insights
        age = patient_data['age']
        if age > 70:
            insights.append({
                'type': 'geriatric_consideration',
                'message': "Geriatric patient - consider age-related pharmacokinetic changes",
                'detail': "Reduced renal/hepatic clearance may require dose adjustments",
                'severity': 'Medium'
            })
        
        return insights
    
    def determine_next_best_actions(self, patient_id):
        """Step 3: Determine Next Best Actions for the Patient"""
        predictions = self.impart_knowledge_and_predict(patient_id)
        patient_data = self.patient_df[self.patient_df['patient_id'] == patient_id].iloc[0]
        
        actions = []
        
        # High priority actions for risk factors
        if predictions['risk_assessment']['risk_level'] == 'High':
            actions.append({
                'action': 'Urgent medication review',
                'priority': 'High',
                'timeline': 'Within 24 hours',
                'reason': f"High overall risk score ({predictions['risk_assessment']['risk_score']:.2f})",
                'responsible': 'Clinical pharmacist + Physician'
            })
        
        # Drug-related actions
        for issue in predictions['drug_suitability']['suitability_issues']:
            if issue['severity'] == 'High':
                actions.append({
                    'action': f"Review {issue['drug_name']} prescription",
                    'priority': 'High',
                    'timeline': 'Within 48 hours',
                    'reason': f"Contraindication or safety concern: {', '.join(issue['issues'])}",
                    'responsible': 'Prescribing physician'
                })
        
        # Care gap actions
        for gap in predictions['care_gaps']:
            if gap['type'] == 'untreated_condition':
                actions.append({
                    'action': f"Consider treatment for {gap['condition']}",
                    'priority': 'Medium',
                    'timeline': 'Within 2 weeks',
                    'reason': gap['message'],
                    'responsible': 'Primary care physician'
                })
            elif gap['type'] == 'monitoring_required':
                actions.append({
                    'action': f"Schedule monitoring for {gap['drug']}",
                    'priority': 'Medium',
                    'timeline': 'Within 1 week',
                    'reason': gap['message'],
                    'responsible': 'Nursing staff + Physician'
                })
        
        # Knowledge insight actions
        for insight in predictions['knowledge_insights']:
            if insight['severity'] == 'High':
                actions.append({
                    'action': 'Address critical medication interaction',
                    'priority': 'High',
                    'timeline': 'Immediate',
                    'reason': insight['detail'],
                    'responsible': 'Physician + Pharmacist'
                })
        
        # Preventive actions based on age and conditions
        age = patient_data['age']
        if age > 65:
            actions.append({
                'action': 'Comprehensive geriatric assessment',
                'priority': 'Medium',
                'timeline': 'Within 1 month',
                'reason': 'Routine geriatric evaluation for optimal medication management',
                'responsible': 'Geriatric team'
            })
        
        # Sort actions by priority
        priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
        actions.sort(key=lambda x: priority_order[x['priority']])
        
        return {
            'patient_id': patient_id,
            'actions': actions,
            'total_actions': len(actions),
            'high_priority_actions': len([a for a in actions if a['priority'] == 'High']),
            'summary': self._generate_action_summary(actions)
        }
    
    def _generate_action_summary(self, actions):
        """Generate executive summary of recommended actions"""
        if not actions:
            return "No specific actions recommended at this time. Continue current management."
        
        high_priority = [a for a in actions if a['priority'] == 'High']
        medium_priority = [a for a in actions if a['priority'] == 'Medium']
        
        summary_parts = []
        
        if high_priority:
            summary_parts.append(f"🚨 {len(high_priority)} high-priority actions requiring immediate attention")
        
        if medium_priority:
            summary_parts.append(f"📋 {len(medium_priority)} medium-priority actions for routine follow-up")
        
        if high_priority:
            urgent_issues = [a['reason'] for a in high_priority[:2]]  # Top 2 urgent issues
            summary_parts.append(f"Key concerns: {'; '.join(urgent_issues)}")
        
        return ". ".join(summary_parts)
    
    def visualize_semantic_net(self, patient_id):
        """Enhanced semantic network visualization with feature categories"""
        self.build_patient_semantic_net(patient_id)
        
        # Create plot with larger figure size
        plt.figure(figsize=(16, 12))
        
        # Define node categories and colors
        category_colors = {
            'patient': '#FF6B6B',      # Red
            'condition': '#4ECDC4',    # Teal
            'drug': '#45B7D1',         # Blue
            'allergy': '#FFA07A',      # Light Salmon
            'organ': '#DA70D6',        # Orchid
            'unknown': '#B0B0B0'       # Gray
        }
        
        # Define node shapes for different categories
        category_shapes = {
            'patient': 's',      # Square
            'condition': 'o',    # Circle
            'drug': 'D',         # Diamond
            'allergy': '^',      # Triangle up
            'organ': 'v',        # Triangle down
            'unknown': 'o'       # Circle
        }
        
        # Define node sizes for different categories
        category_sizes = {
            'patient': 2000,
            'condition': 1200,
            'drug': 1000,
            'allergy': 800,
            'organ': 1000,
            'unknown': 600
        }
        
        # Prepare data for plotting
        node_colors = []
        node_sizes = []
        node_labels = {}
        node_categories = []
        
        for node in self.semantic_net.nodes():
            node_type = self.semantic_net.nodes[node].get('type', 'unknown')
            node_colors.append(category_colors.get(node_type, category_colors['unknown']))
            node_sizes.append(category_sizes.get(node_type, category_sizes['unknown']))
            node_categories.append(node_type)
            
            # Create labels
            if node_type == 'patient':
                age = self.semantic_net.nodes[node].get('age', '')
                gender = self.semantic_net.nodes[node].get('gender', '')
                node_labels[node] = f"Patient\nAge: {age}\nGender: {gender}"
            elif node_type == 'condition':
                name = self.semantic_net.nodes[node].get('name', node.split('_')[-1])
                node_labels[node] = f"Condition:\n{name}"
            elif node_type == 'drug':
                name = self.semantic_net.nodes[node].get('name', node.split('_')[-1])
                current = self.semantic_net.nodes[node].get('current', False)
                status = "Current" if current else "Potential"
                dosage = self.semantic_net.nodes[node].get('dosage', '')
                label = f"Drug ({status}):\n{name}"
                if dosage:
                    label += f"\n{dosage}"
                node_labels[node] = label
            elif node_type == 'allergy':
                name = self.semantic_net.nodes[node].get('name', node.split('_')[-1])
                node_labels[node] = f"Allergy:\n{name}"
            elif node_type == 'organ':
                name = self.semantic_net.nodes[node].get('name', '')
                function = self.semantic_net.nodes[node].get('function', '')
                node_labels[node] = f"Organ:\n{name}\n({function})"
            else:
                node_labels[node] = node.split('_')[-1]
        
        # Define layout with better spacing
        pos = nx.spring_layout(self.semantic_net, k=2, iterations=100, seed=42)
        
        # Create legend elements
        unique_categories = list(set(node_categories))
        legend_elements = []
        for category in unique_categories:
            color = category_colors.get(category, category_colors['unknown'])
            shape = category_shapes.get(category, category_shapes['unknown'])
            legend_elements.append(plt.Line2D([0], [0], marker=shape, color='w', 
                                            markerfacecolor=color, markersize=10, 
                                            label=category.title()))
        
        # Draw edges first
        nx.draw_networkx_edges(self.semantic_net, pos, alpha=0.6, edge_color='gray', 
                              width=1.5, style='dashed')
        
        # Draw nodes by category for better legend representation
        for category in unique_categories:
            category_nodes = [node for i, node in enumerate(self.semantic_net.nodes()) 
                             if node_categories[i] == category]
            
            if category_nodes:
                node_color = category_colors.get(category, category_colors['unknown'])
                node_shape = category_shapes.get(category, category_shapes['unknown'])
                node_size = category_sizes.get(category, category_sizes['unknown'])
                
                nx.draw_networkx_nodes(self.semantic_net, pos, nodelist=category_nodes,
                                     node_color=node_color, node_shape=node_shape,
                                     node_size=node_size, alpha=0.9, 
                                     edgecolors='black', linewidths=2)
        
        # Add labels with better formatting
        nx.draw_networkx_labels(self.semantic_net, pos, node_labels, font_size=8, 
                               font_weight='bold', font_family='sans-serif')
        
        # Add edge labels for relationships
        edge_labels = nx.get_edge_attributes(self.semantic_net, 'relationship')
        nx.draw_networkx_edge_labels(self.semantic_net, pos, edge_labels, 
                                    font_size=7, font_color='darkblue')
        
        plt.title(f"Semantic Network for Patient {patient_id}\nFeature Categories Visualization", 
                  size=16, pad=20, weight='bold')
        
        # Add legend
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1),
                   frameon=True, fancybox=True, shadow=True, ncol=1,
                   title="Feature Categories", title_fontsize=12)
        
        plt.axis('off')
        
        # Add summary statistics
        summary_text = f"""
        Network Summary:
        • Total Nodes: {len(self.semantic_net.nodes())}
        • Total Edges: {len(self.semantic_net.edges())}
        • Categories: {', '.join(sorted(unique_categories))}
        """
        
        plt.figtext(0.02, 0.02, summary_text, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        return plt

# ===============================
# 1. PATIENT DATA MANAGEMENT WITH STORAGE
# ===============================

class PatientDataManager:
    def __init__(self):
        self.storage_manager = DataStorageManager()
        self.common_conditions = [
            'Hypertension', 'Diabetes', 'Asthma', 'Heart Disease', 'Kidney Disease',
            'Liver Disease', 'COPD', 'Arthritis', 'Depression', 'Anxiety',
            'Hyperlipidemia', 'Thyroid Disorder', 'Osteoporosis', 'Cancer',
            'HIV/AIDS', 'Epilepsy', 'Parkinson\'s', 'Alzheimer\'s', 'Stroke',
            'Heart Failure', 'Chronic Pain', 'Migraine', 'Obesity', 'Anemia'
        ]
        
        self.common_allergies = [
            'Penicillin', 'Sulfa Drugs', 'NSAIDs', 'Aspirin', 'Codeine',
            'Morphine', 'Iodine', 'Latex', 'Peanuts', 'Shellfish',
            'Eggs', 'Milk', 'Soy', 'Wheat', 'Tree Nuts', 'Dust Mites',
            'Pollen', 'Mold', 'Animal Dander', 'Insect Stings'
        ]
        
        self.common_drugs = [
            'Lisinopril', 'Metformin', 'Warfarin', 'Atorvastatin', 'Metoprolol',
            'Amlodipine', 'Levothyroxine', 'Albuterol', 'Sertraline', 'Omeprazole',
            'Simvastatin', 'Losartan', 'Gabapentin', 'Hydrochlorothiazide', 'Furosemide',
            'Insulin', 'Aspirin', 'Ibuprofen', 'Acetaminophen', 'Prednisone',
            'Warfarin', 'Clopidogrel', 'Digoxin', 'Spironolactone', 'Carvedilol'
        ]
        
        self.first_names = [
            'James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda',
            'William', 'Elizabeth', 'David', 'Barbara', 'Richard', 'Susan', 'Joseph', 'Jessica',
            'Thomas', 'Sarah', 'Charles', 'Karen', 'Christopher', 'Nancy', 'Daniel', 'Lisa',
            'Matthew', 'Betty', 'Anthony', 'Margaret', 'Mark', 'Sandra', 'Donald', 'Ashley',
            'Steven', 'Kimberly', 'Paul', 'Emily', 'Andrew', 'Donna', 'Joshua', 'Michelle',
            'Kenneth', 'Carol', 'Kevin', 'Amanda', 'Brian', 'Dorothy', 'George', 'Melissa',
            'Timothy', 'Deborah', 'Ronald', 'Stephanie', 'Jason', 'Rebecca', 'Edward', 'Sharon',
            'Jeffrey', 'Laura', 'Ryan', 'Cynthia', 'Jacob', 'Kathleen', 'Gary', 'Amy', 'Nicholas',
            'Angela', 'Eric', 'Shirley', 'Jonathan', 'Brenda', 'Stephen', 'Pamela', 'Larry', 'Emma'
        ]
        
        self.last_names = [
            'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
            'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
            'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson',
            'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker',
            'Young', 'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores',
            'Green', 'Adams', 'Nelson', 'Baker', 'Hall', 'Rivera', 'Campbell', 'Mitchell',
            'Carter', 'Roberts'
        ]
        
        self.doctors = [
            'Dr. Wilson', 'Dr. Garcia', 'Dr. Thompson', 'Dr. Lee', 'Dr. Martinez',
            'Dr. Anderson', 'Dr. Taylor', 'Dr. Moore', 'Dr. Jackson', 'Dr. Martin',
            'Dr. Harris', 'Dr. Clark', 'Dr. Lewis', 'Dr. Walker', 'Dr. Hall',
            'Dr. Young', 'Dr. King', 'Dr. Wright', 'Dr. Scott', 'Dr. Green'
        ]
    
    def generate_large_dataset(self):
        """Generate a dataset with 100+ patients"""
        np.random.seed(42)
        
        patient_data = []
        prescription_data = []
        
        # Generate 120 patients
        for i in range(1, 121):
            patient_id = f"P{1000 + i}"
            first_name = np.random.choice(self.first_names)
            last_name = np.random.choice(self.last_names)
            age = np.random.randint(18, 95)
            gender = np.random.choice(['Male', 'Female'])
            blood_type = np.random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'])
            weight_kg = np.random.randint(45, 120)
            height_cm = np.random.randint(150, 195)
            
            # Generate medical conditions (1-4 conditions per patient)
            num_conditions = np.random.randint(1, 5)
            medical_conditions = list(np.random.choice(self.common_conditions, num_conditions, replace=False))
            
            # Generate allergies (0-3 allergies per patient)
            num_allergies = np.random.randint(0, 4)
            allergies = list(np.random.choice(self.common_allergies, num_allergies, replace=False)) if num_allergies > 0 else ['None']
            
            # Organ function with age correlation
            kidney_options = ['normal', 'mild', 'moderate', 'severe']
            kidney_weights = [0.7, 0.15, 0.1, 0.05]
            if age > 70:
                kidney_weights = [0.4, 0.3, 0.2, 0.1]
            kidney_function = np.random.choice(kidney_options, p=kidney_weights)
            
            liver_options = ['normal', 'mild', 'moderate', 'severe']
            liver_weights = [0.8, 0.1, 0.07, 0.03]
            liver_function = np.random.choice(liver_options, p=liver_weights)
            
            # ADR history (higher probability for older patients)
            adr_prob = min(0.3 + (age - 50) * 0.01, 0.6) if age > 50 else 0.1
            previous_adr_history = np.random.choice([0, 1], p=[1-adr_prob, adr_prob])
            
            # Comorbidity score based on age and number of conditions
            comorbidity_score = min(len(medical_conditions) + (age // 25), 10)
            
            emergency_contact = f"Contact {i} - 555-{1000 + i:04d}"
            primary_physician = np.random.choice(self.doctors)
            
            # Admission date in the last 2 years
            days_ago = np.random.randint(1, 730)
            admission_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
            
            # Room number
            floor = np.random.randint(1, 5)
            room = np.random.randint(101, 150)
            wing = np.random.choice(['A', 'B', 'C', 'D'])
            room_number = f"{floor}{room:02d}{wing}"
            
            patient_data.append({
                'patient_id': patient_id,
                'first_name': first_name,
                'last_name': last_name,
                'age': age,
                'gender': gender,
                'blood_type': blood_type,
                'weight_kg': weight_kg,
                'height_cm': height_cm,
                'medical_conditions': medical_conditions,
                'allergies': allergies,
                'kidney_function': kidney_function,
                'liver_function': liver_function,
                'previous_adr_history': previous_adr_history,
                'comorbidity_score': comorbidity_score,
                'emergency_contact': emergency_contact,
                'primary_physician': primary_physician,
                'admission_date': admission_date,
                'room_number': room_number
            })
            
            # Generate prescriptions for this patient (1-6 medications)
            num_prescriptions = np.random.randint(1, 7)
            patient_drugs = list(np.random.choice(self.common_drugs, num_prescriptions, replace=False))
            
            for drug in patient_drugs:
                # Dosage based on drug type
                if drug in ['Warfarin', 'Digoxin']:
                    dosage = f"{np.random.choice([2.5, 5, 7.5])}mg daily"
                elif drug in ['Metformin', 'Gabapentin']:
                    dosage = f"{np.random.choice([500, 750, 1000])}mg {'twice daily' if np.random.random() > 0.5 else 'daily'}"
                elif drug in ['Atorvastatin', 'Simvastatin']:
                    dosage = f"{np.random.choice([10, 20, 40, 80])}mg daily"
                elif drug in ['Lisinopril', 'Losartan']:
                    dosage = f"{np.random.choice([5, 10, 20, 40])}mg daily"
                else:
                    dosage = f"{np.random.choice([25, 50, 100])}mg {'twice daily' if np.random.random() > 0.5 else 'daily'}"
                
                start_days_ago = np.random.randint(0, days_ago)
                start_date = (datetime.now() - timedelta(days=start_days_ago)).strftime('%Y-%m-%d')
                
                frequency_options = ['Once daily', 'Twice daily', 'Three times daily', 'Four times daily', 'As needed']
                frequency = np.random.choice(frequency_options, p=[0.6, 0.2, 0.1, 0.05, 0.05])
                
                route_options = ['Oral', 'IV', 'IM', 'Subcutaneous', 'Topical', 'Inhalation']
                route_weights = [0.8, 0.05, 0.05, 0.05, 0.03, 0.02]
                route = np.random.choice(route_options, p=route_weights)
                
                prescription_data.append({
                    'patient_id': patient_id,
                    'drug_name': drug,
                    'dosage': dosage,
                    'start_date': start_date,
                    'prescribing_doctor': primary_physician,
                    'frequency': frequency,
                    'route': route
                })
        
        return pd.DataFrame(patient_data), pd.DataFrame(prescription_data)
    
    def initialize_sample_data(self):
        """Initialize sample patient and prescription data"""
        return self.generate_large_dataset()
    
    def save_data(self, patient_df, prescription_df):
        """Save both patient and prescription data"""
        success1 = self.storage_manager.save_patients(patient_df)
        success2 = self.storage_manager.save_prescriptions(prescription_df)
        return success1 and success2
    
    def load_data(self):
        """Load both patient and prescription data"""
        patient_df = self.storage_manager.load_patients()
        prescription_df = self.storage_manager.load_prescriptions()
        return patient_df, prescription_df
    
    def add_new_patient(self, patient_data, prescription_data):
        """Add a new patient to the system"""
        st.subheader("➕ Add New Patient")
        
        with st.form("patient_form"):
            st.markdown("<div class='form-container'>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                patient_id = st.text_input("Patient ID*", value=f"P{1000 + len(patient_data) + 1}")
                first_name = st.text_input("First Name*")
                last_name = st.text_input("Last Name*")
                age = st.number_input("Age*", min_value=0, max_value=120, value=50)
                gender = st.selectbox("Gender*", ["Male", "Female", "Other"])
                blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"])
                
            with col2:
                weight_kg = st.number_input("Weight (kg)", min_value=0.0, value=70.0)
                height_cm = st.number_input("Height (cm)", min_value=0.0, value=170.0)
                kidney_function = st.selectbox("Kidney Function", ["normal", "mild", "moderate", "severe"])
                liver_function = st.selectbox("Liver Function", ["normal", "mild", "moderate", "severe"])
                previous_adr = st.selectbox("Previous ADR History", [0, 1])
                comorbidity_score = st.slider("Comorbidity Score", 1, 10, 3)
            
            # Medical Conditions
            st.subheader("Medical Conditions")
            selected_conditions = st.multiselect("Select conditions:", self.common_conditions)
            other_conditions = st.text_input("Other conditions (comma-separated):")
            
            # Allergies
            st.subheader("Allergies")
            selected_allergies = st.multiselect("Select allergies:", self.common_allergies)
            other_allergies = st.text_input("Other allergies (comma-separated):")
            
            # Contact Information
            st.subheader("Contact Information")
            emergency_contact = st.text_input("Emergency Contact")
            primary_physician = st.text_input("Primary Physician")
            room_number = st.text_input("Room Number")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            submitted = st.form_submit_button("Add Patient")
            
            if submitted:
                if not first_name or not last_name:
                    st.error("Please fill in all required fields (*)")
                    return patient_data, prescription_data
                
                # Combine selected and other conditions/allergies
                all_conditions = selected_conditions
                if other_conditions:
                    all_conditions.extend([cond.strip() for cond in other_conditions.split(',')])
                
                all_allergies = selected_allergies
                if other_allergies:
                    all_allergies.extend([allergy.strip() for allergy in other_allergies.split(',')])
                
                # Add new patient
                new_patient = {
                    'patient_id': patient_id,
                    'first_name': first_name,
                    'last_name': last_name,
                    'age': age,
                    'gender': gender,
                    'blood_type': blood_type,
                    'weight_kg': weight_kg,
                    'height_cm': height_cm,
                    'medical_conditions': all_conditions,
                    'allergies': all_allergies,
                    'kidney_function': kidney_function,
                    'liver_function': liver_function,
                    'previous_adr_history': previous_adr,
                    'comorbidity_score': comorbidity_score,
                    'emergency_contact': emergency_contact,
                    'primary_physician': primary_physician,
                    'admission_date': datetime.now().strftime('%Y-%m-%d'),
                    'room_number': room_number
                }
                
                # Add to patient data
                patient_data = pd.concat([patient_data, pd.DataFrame([new_patient])], ignore_index=True)
                st.success(f"Patient {first_name} {last_name} added successfully!")
                
                return patient_data, prescription_data
        
        return patient_data, prescription_data
    
    def delete_patient(self, patient_id, patient_data, prescription_data):
        """Delete a patient and their prescriptions"""
        # Remove patient from patient data
        patient_data = patient_data[patient_data['patient_id'] != patient_id]
        
        # Remove patient's prescriptions
        prescription_data = prescription_data[prescription_data['patient_id'] != patient_id]
        
        st.success(f"Patient {patient_id} and their prescriptions have been deleted.")
        return patient_data, prescription_data
    
    def edit_patient(self, patient_id, patient_data):
        """Edit an existing patient's information"""
        st.subheader(f"✏️ Edit Patient: {patient_id}")
        
        # Find patient data
        patient_row = patient_data[patient_data['patient_id'] == patient_id]
        if len(patient_row) == 0:
            st.error(f"Patient {patient_id} not found.")
            return patient_data
        
        patient_info = patient_row.iloc[0]
        
        with st.form("edit_patient_form"):
            st.markdown("<div class='form-container'>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                first_name = st.text_input("First Name*", value=patient_info.get('first_name', ''))
                last_name = st.text_input("Last Name*", value=patient_info.get('last_name', ''))
                age = st.number_input("Age*", min_value=0, max_value=120, value=int(patient_info.get('age', 50)))
                gender = st.selectbox("Gender*", ["Male", "Female", "Other"], 
                                    index=["Male", "Female", "Other"].index(patient_info.get('gender', 'Male')))
                blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"],
                                        index=["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"].index(patient_info.get('blood_type', 'Unknown')))
                
            with col2:
                weight_kg = st.number_input("Weight (kg)", min_value=0.0, value=float(patient_info.get('weight_kg', 70.0)))
                height_cm = st.number_input("Height (cm)", min_value=0.0, value=float(patient_info.get('height_cm', 170.0)))
                kidney_function = st.selectbox("Kidney Function", ["normal", "mild", "moderate", "severe"],
                                            index=["normal", "mild", "moderate", "severe"].index(patient_info.get('kidney_function', 'normal')))
                liver_function = st.selectbox("Liver Function", ["normal", "mild", "moderate", "severe"],
                                            index=["normal", "mild", "moderate", "severe"].index(patient_info.get('liver_function', 'normal')))
                previous_adr = st.selectbox("Previous ADR History", [0, 1], 
                                          index=[0, 1].index(patient_info.get('previous_adr_history', 0)))
                comorbidity_score = st.slider("Comorbidity Score", 1, 10, int(patient_info.get('comorbidity_score', 3)))
            
            # Medical Conditions
            st.subheader("Medical Conditions")
            current_conditions = patient_info.get('medical_conditions', [])
            selected_conditions = st.multiselect("Select conditions:", self.common_conditions, default=current_conditions)
            other_conditions = st.text_input("Other conditions (comma-separated):", 
                                           value=", ".join([c for c in current_conditions if c not in self.common_conditions]))
            
            # Allergies
            st.subheader("Allergies")
            current_allergies = patient_info.get('allergies', [])
            selected_allergies = st.multiselect("Select allergies:", self.common_allergies, default=current_allergies)
            other_allergies = st.text_input("Other allergies (comma-separated):", 
                                          value=", ".join([a for a in current_allergies if a not in self.common_allergies]))
            
            # Contact Information
            st.subheader("Contact Information")
            emergency_contact = st.text_input("Emergency Contact", value=patient_info.get('emergency_contact', ''))
            primary_physician = st.text_input("Primary Physician", value=patient_info.get('primary_physician', ''))
            room_number = st.text_input("Room Number", value=patient_info.get('room_number', ''))
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            submitted = st.form_submit_button("Update Patient")
            
            if submitted:
                if not first_name or not last_name:
                    st.error("Please fill in all required fields (*)")
                    return patient_data
                
                # Combine selected and other conditions/allergies
                all_conditions = selected_conditions
                if other_conditions:
                    all_conditions.extend([cond.strip() for cond in other_conditions.split(',') if cond.strip()])
                
                all_allergies = selected_allergies
                if other_allergies:
                    all_allergies.extend([allergy.strip() for allergy in other_allergies.split(',') if allergy.strip()])
                
                # Create updated patient record
                updated_patient = {
                    'patient_id': patient_id,
                    'first_name': first_name,
                    'last_name': last_name,
                    'age': age,
                    'gender': gender,
                    'blood_type': blood_type,
                    'weight_kg': weight_kg,
                    'height_cm': height_cm,
                    'medical_conditions': all_conditions,
                    'allergies': all_allergies,
                    'kidney_function': kidney_function,
                    'liver_function': liver_function,
                    'previous_adr_history': previous_adr,
                    'comorbidity_score': comorbidity_score,
                    'emergency_contact': emergency_contact,
                    'primary_physician': primary_physician,
                    'admission_date': patient_info.get('admission_date', datetime.now().strftime('%Y-%m-%d')),
                    'room_number': room_number
                }
                
                # Remove the old record and add the updated one
                patient_data = patient_data[patient_data['patient_id'] != patient_id]
                patient_data = pd.concat([patient_data, pd.DataFrame([updated_patient])], ignore_index=True)
                
                st.success(f"Patient {first_name} {last_name} updated successfully!")
                return patient_data
        
        return patient_data
    
    def add_prescription(self, patient_id, prescription_data):
        """Add prescription for a patient"""
        st.subheader("💊 Add Prescription")
        
        with st.form("prescription_form"):
            st.markdown("<div class='form-container'>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                drug_name = st.selectbox("Drug Name*", self.common_drugs)
                dosage = st.text_input("Dosage*", placeholder="e.g., 10mg daily")
                start_date = st.date_input("Start Date*", value=datetime.now())
                frequency = st.selectbox("Frequency*", ["Once daily", "Twice daily", "Three times daily", "Four times daily", "As needed"])
                
            with col2:
                route = st.selectbox("Route*", ["Oral", "IV", "IM", "Subcutaneous", "Topical", "Inhalation"])
                prescribing_doctor = st.text_input("Prescribing Doctor*")
                end_date = st.date_input("End Date (optional)", value=None)
                instructions = st.text_area("Special Instructions")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            submitted = st.form_submit_button("Add Prescription")
            
            if submitted:
                if not drug_name or not dosage or not prescribing_doctor:
                    st.error("Please fill in all required fields (*)")
                    return prescription_data
                
                new_prescription = {
                    'patient_id': patient_id,
                    'drug_name': drug_name,
                    'dosage': dosage,
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'prescribing_doctor': prescribing_doctor,
                    'frequency': frequency,
                    'route': route,
                    'end_date': end_date.strftime('%Y-%m-%d') if end_date else None,
                    'instructions': instructions
                }
                
                prescription_data = pd.concat([prescription_data, pd.DataFrame([new_prescription])], ignore_index=True)
                st.success(f"Prescription for {drug_name} added successfully!")
                
                return prescription_data
        
        return prescription_data

    def get_patient_statistics(self, patient_df):
        """Generate statistics about the patient population"""
        stats = {
            'total_patients': len(patient_df),
            'avg_age': patient_df['age'].mean(),
            'gender_distribution': patient_df['gender'].value_counts().to_dict(),
            'common_conditions': self._get_common_items(patient_df, 'medical_conditions'),
            'common_allergies': self._get_common_items(patient_df, 'allergies'),
            'age_groups': self._get_age_groups(patient_df),
            'comorbidity_stats': {
                'min': patient_df['comorbidity_score'].min(),
                'max': patient_df['comorbidity_score'].max(),
                'avg': patient_df['comorbidity_score'].mean()
            }
        }
        return stats
    
    def _get_common_items(self, df, column):
        """Get most common items from list columns"""
        all_items = []
        for items in df[column]:
            if isinstance(items, list):
                all_items.extend(items)
        from collections import Counter
        return Counter(all_items).most_common(10)
    
    def _get_age_groups(self, df):
        """Categorize patients by age groups"""
        bins = [0, 18, 30, 45, 60, 75, 100]
        labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76+']
        return pd.cut(df['age'], bins=bins, labels=labels).value_counts().to_dict()

# ===============================
# 2. REAL DATABASE INTEGRATION
# ===============================

class RealDrugDatabase:
    def __init__(self):
        self.fda_api_base = "https://api.fda.gov/drug"
        self.drugbank_api_base = "https://go.drugbank.com"
        self.interactions_cache = {}
        
    def get_fda_drug_interactions(self, drug_name):
        """Fetch real drug interactions from FDA API"""
        try:
            url = f"{self.fda_api_base}/label.json"
            params = {
                'search': f'openfda.brand_name:"{drug_name}"',
                'limit': 1
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return self._parse_fda_interactions(data)
        except Exception as e:
            st.warning(f"FDA API Error for {drug_name}: Using fallback data")
            return self._get_fallback_interactions(drug_name)
        return []
    
    def get_drugbank_interactions(self, drug_name):
        """Fetch from DrugBank (mock - would require API key in production)"""
        mock_interactions = {
            'Warfarin': ['Aspirin', 'Ibuprofen', 'Naproxen', 'SSRI antidepressants'],
            'Metformin': ['Contrast agents', 'Alcohol', 'Corticosteroids'],
            'Lisinopril': ['Potassium supplements', 'NSAIDs', 'Diuretics'],
            'Atorvastatin': ['Antifungals', 'Macrolide antibiotics', 'Grapefruit juice'],
            'Metoprolol': ['Calcium channel blockers', 'Insulin', 'Clonidine']
        }
        return mock_interactions.get(drug_name, [])
    
    def _parse_fda_interactions(self, data):
        interactions = []
        if 'results' in data and len(data['results']) > 0:
            drug_info = data['results'][0]
            if 'drug_interactions' in drug_info:
                interactions.extend(drug_info['drug_interactions'])
            elif 'warnings' in drug_info:
                interactions.extend(drug_info['warnings'])
        return interactions
    
    def _get_fallback_interactions(self, drug_name):
        fallback_db = {
            'Warfarin': ['Increased bleeding risk with NSAIDs, antiplatelets'],
            'Metformin': ['Lactic acidosis risk with contrast media'],
            'Insulin': ['Hypoglycemia with beta-blockers, alcohol'],
            'Digoxin': ['Toxicity with diuretics, calcium channel blockers']
        }
        return fallback_db.get(drug_name, [])

# ===============================
# 3. MACHINE LEARNING FOR ADVERSE DRUG REACTIONS
# ===============================

class ADRPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.features = ['age', 'comorbidity_score', 'num_medications', 
                        'kidney_function', 'liver_function', 'previous_adr']
        self.is_trained = False
        
    def generate_training_data(self, n_samples=5000):
        np.random.seed(42)
        
        data = {
            'age': np.random.randint(18, 95, n_samples),
            'comorbidity_score': np.random.randint(1, 10, n_samples),
            'num_medications': np.random.randint(1, 15, n_samples),
            'kidney_function': np.random.choice(['normal', 'mild', 'moderate', 'severe'], n_samples, p=[0.7, 0.15, 0.1, 0.05]),
            'liver_function': np.random.choice(['normal', 'mild', 'moderate', 'severe'], n_samples, p=[0.8, 0.1, 0.07, 0.03]),
            'previous_adr': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
        }
        
        df = pd.DataFrame(data)
        
        df['kidney_function'] = df['kidney_function'].map({'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 3})
        df['liver_function'] = df['liver_function'].map({'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 3})
        
        adr_risk = (
            df['age'] * 0.01 +
            df['comorbidity_score'] * 0.1 +
            df['num_medications'] * 0.08 +
            df['kidney_function'] * 0.15 +
            df['liver_function'] * 0.12 +
            df['previous_adr'] * 0.3 +
            np.random.normal(0, 0.1, n_samples)
        )
        
        df['adr_probability'] = 1 / (1 + np.exp(-adr_risk))
        df['adr_occurred'] = (df['adr_probability'] > 0.6).astype(int)
        
        return df
    
    def train_model(self):
        with st.spinner("Training ADR Prediction Model..."):
            data = self.generate_training_data(2000)
            
            X = data[self.features]
            y = data['adr_occurred']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.model.fit(X_train, y_train)
            
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            return accuracy
    
    def predict_adr_risk(self, patient_data, current_medications):
        if not self.is_trained:
            accuracy = self.train_model()
            st.sidebar.success(f"ADR Model trained with accuracy: {accuracy:.2%}")
        
        features = {
            'age': patient_data['age'],
            'comorbidity_score': patient_data.get('comorbidity_score', 1),
            'num_medications': len(current_medications),
            'kidney_function': patient_data.get('kidney_function', 'normal'),
            'liver_function': patient_data.get('liver_function', 'normal'),
            'previous_adr': patient_data.get('previous_adr_history', 0)
        }
        
        kidney_map = {'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
        liver_map = {'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
        
        features['kidney_function'] = kidney_map.get(features['kidney_function'], 0)
        features['liver_function'] = liver_map.get(features['liver_function'], 0)
        
        X = np.array([[features[feat] for feat in self.features]])
        
        probability = self.model.predict_proba(X)[0, 1]
        risk_level = 'Low' if probability < 0.3 else 'Medium' if probability < 0.7 else 'High'
        
        return {
            'probability': probability,
            'risk_level': risk_level,
            'factors_contributing': self._get_risk_factors(features, probability)
        }
    
    def _get_risk_factors(self, features, probability):
        factors = []
        if features['age'] > 65:
            factors.append("Advanced age")
        if features['comorbidity_score'] > 5:
            factors.append("Multiple comorbidities")
        if features['num_medications'] > 5:
            factors.append("Polypharmacy (multiple medications)")
        if features['kidney_function'] > 1:
            factors.append("Impaired kidney function")
        if features['liver_function'] > 1:
            factors.append("Impaired liver function")
        if features['previous_adr'] == 1:
            factors.append("Previous adverse drug reaction history")
            
        return factors

# ===============================
# 4. REAL-TIME MONITORING SYSTEM
# ===============================

class RealTimeMonitor:
    def __init__(self, analyzer, alert_system):
        self.analyzer = analyzer
        self.alert_system = alert_system
        self.monitored_patients = {}
        self.vital_signs_thresholds = {
            'heart_rate': (60, 100),
            'blood_pressure_systolic': (90, 140),
            'blood_pressure_diastolic': (60, 90),
            'temperature': (36.1, 37.8),
            'respiratory_rate': (12, 20)
        }
    
    def add_patient_for_monitoring(self, patient_id, analysis_result):
        self.monitored_patients[patient_id] = {
            'analysis': analysis_result,
            'last_vitals_check': datetime.now(),
            'alerts_triggered': [],
            'vital_signs_history': []
        }
    
    def simulate_vital_signs_update(self, patient_id):
        if patient_id not in self.monitored_patients:
            return None
            
        vitals = {
            'timestamp': datetime.now(),
            'heart_rate': np.random.randint(50, 120),
            'blood_pressure_systolic': np.random.randint(80, 180),
            'blood_pressure_diastolic': np.random.randint(50, 110),
            'temperature': round(np.random.uniform(36.0, 39.0), 1),
            'respiratory_rate': np.random.randint(10, 25),
            'oxygen_saturation': np.random.randint(85, 100)
        }
        
        self.monitored_patients[patient_id]['vital_signs_history'].append(vitals)
        self._check_vital_signs_alerts(patient_id, vitals)
        
        return vitals
    
    def _check_vital_signs_alerts(self, patient_id, vitals):
        alerts = []
        patient_analysis = self.monitored_patients[patient_id]['analysis']
        
        for sign, (low, high) in self.vital_signs_thresholds.items():
            value = vitals[sign]
            if value < low:
                alerts.append(f"Low {sign.replace('_', ' ')}: {value} (normal: {low}-{high})")
            elif value > high:
                alerts.append(f"High {sign.replace('_', ' ')}: {value} (normal: {low}-{high})")
        
        for drug_analysis in patient_analysis['prescriptions']:
            drug_name = drug_analysis['drug_name']
            if drug_name == 'Warfarin' and vitals['heart_rate'] > 100:
                alerts.append("Elevated heart rate - monitor for warfarin complications")
            if drug_name in ['Metoprolol', 'Lisinopril'] and vitals['blood_pressure_systolic'] < 100:
                alerts.append("Low blood pressure - review antihypertensive medication")
        
        for alert in alerts:
            if alert not in self.monitored_patients[patient_id]['alerts_triggered']:
                self.alert_system.send_alert(
                    patient_id=patient_id,
                    alert_type="Vital Signs Alert",
                    message=alert,
                    severity="Medium"
                )
                self.monitored_patients[patient_id]['alerts_triggered'].append(alert)

# ===============================
# 5. ALERT SYSTEM
# ===============================

class AlertSystem:
    def __init__(self):
        self.sent_alerts = []
    
    def send_alert(self, patient_id, alert_type, message, severity="High"):
        alert = {
            'timestamp': datetime.now(),
            'patient_id': patient_id,
            'alert_type': alert_type,
            'message': message,
            'severity': severity,
            'acknowledged': False
        }
        
        self.sent_alerts.append(alert)
    
    def get_pending_alerts(self):
        return [alert for alert in self.sent_alerts if not alert['acknowledged']]
    
    def acknowledge_alert(self, alert_index):
        if 0 <= alert_index < len(self.sent_alerts):
            self.sent_alerts[alert_index]['acknowledged'] = True

# ===============================
# 6. ENHANCED DRUG ANALYZER WITH ALL FEATURES
# ===============================

class EnhancedDrugAnalyzer:
    def __init__(self, patient_df, prescription_df):
        self.patient_df = patient_df
        self.prescription_df = prescription_df
        self.real_db = RealDrugDatabase()
        self.adr_predictor = ADRPredictor()
        self.alert_system = AlertSystem()
        self.monitor = RealTimeMonitor(self, self.alert_system)
        self.knowledge_agent = KnowledgeBasedAIAgent(patient_df, prescription_df)
        
    def comprehensive_patient_analysis(self, patient_id):
        patient_data = self.patient_df[self.patient_df['patient_id'] == patient_id].iloc[0]
        prescriptions = self.prescription_df[self.prescription_df['patient_id'] == patient_id]
        
        analysis_results = {
            'patient_info': patient_data.to_dict(),
            'prescriptions': [],
            'real_database_interactions': [],
            'adr_prediction': {},
            'vital_signs': {},
            'alerts': [],
            'knowledge_analysis': self.knowledge_agent.impart_knowledge_and_predict(patient_id),
            'next_best_actions': self.knowledge_agent.determine_next_best_actions(patient_id)
        }
        
        for _, prescription in prescriptions.iterrows():
            drug_analysis = self._analyze_drug_with_real_data(patient_data, prescription)
            analysis_results['prescriptions'].append(drug_analysis)
        
        analysis_results['real_database_interactions'] = self._check_real_drug_interactions(prescriptions)
        
        current_meds = prescriptions['drug_name'].tolist()
        analysis_results['adr_prediction'] = self.adr_predictor.predict_adr_risk(
            patient_data, current_meds
        )
        
        self.monitor.add_patient_for_monitoring(patient_id, analysis_results)
        self._generate_initial_alerts(patient_id, analysis_results)
        
        return analysis_results
    
    def _analyze_drug_with_real_data(self, patient_data, prescription):
        drug_name = prescription['drug_name']
        
        analysis = {
            'drug_name': drug_name,
            'dosage': prescription['dosage'],
            'fda_interactions': self.real_db.get_fda_drug_interactions(drug_name),
            'drugbank_interactions': self.real_db.get_drugbank_interactions(drug_name),
            'contraindications': self._check_real_contraindications(patient_data, drug_name),
            'monitoring_requirements': self._get_monitoring_requirements(drug_name)
        }
        
        return analysis
    
    def _check_real_drug_interactions(self, prescriptions):
        interactions = []
        drug_list = prescriptions['drug_name'].tolist()
        
        for i, drug1 in enumerate(drug_list):
            for drug2 in drug_list[i+1:]:
                fda_interactions = self.real_db.get_fda_drug_interactions(drug1)
                drugbank_interactions = self.real_db.get_drugbank_interactions(drug1)
                
                if drug2 in drugbank_interactions:
                    interactions.append({
                        'drug1': drug1,
                        'drug2': drug2,
                        'source': 'DrugBank',
                        'risk_level': 'High',
                        'message': f'Known interaction between {drug1} and {drug2}'
                    })
        
        return interactions
    
    def _check_real_contraindications(self, patient_data, drug_name):
        contraindications = []
        conditions = patient_data['medical_conditions']
        
        contraindication_rules = {
            'Warfarin': ['Liver_Disease', 'Bleeding_Disorders', 'Pregnancy'],
            'Metformin': ['Severe_Kidney_Disease', 'Liver_Failure'],
            'Lisinopril': ['Pregnancy', 'Angioedema_History', 'Kidney_Disease'],
            'Atorvastatin': ['Active_Liver_Disease', 'Pregnancy'],
            'Metoprolol': ['Asthma', 'Heart_Block']
        }
        
        for condition in conditions:
            if drug_name in contraindication_rules and condition in contraindication_rules[drug_name]:
                contraindications.append({
                    'condition': condition,
                    'risk_level': 'High',
                    'message': f'{drug_name} contraindicated in {condition}'
                })
        
        return contraindications
    
    def _get_monitoring_requirements(self, drug_name):
        monitoring_map = {
            'Warfarin': ['INR weekly', 'CBC monthly', 'Watch for bleeding'],
            'Diuretics': ['Electrolytes monthly', 'Renal function', 'Blood pressure'],
            'Lisinopril': ['Renal function', 'Potassium levels', 'Blood pressure'],
            'Atorvastatin': ['Liver enzymes', 'CPK levels', 'Blood glucose']
        }
        return monitoring_map.get(drug_name, ['Routine monitoring'])
    
    def _generate_initial_alerts(self, patient_id, analysis):
        if analysis['adr_prediction']['risk_level'] == 'High':
            self.alert_system.send_alert(
                patient_id=patient_id,
                alert_type="High ADR Risk",
                message=f"Patient has {analysis['adr_prediction']['probability']:.1%} probability of adverse drug reaction",
                severity="High"
            )
        
        for interaction in analysis['real_database_interactions']:
            self.alert_system.send_alert(
                patient_id=patient_id,
                alert_type="Drug Interaction",
                message=interaction['message'],
                severity=interaction['risk_level']
            )
        
        for drug in analysis['prescriptions']:
            for contra in drug['contraindications']:
                self.alert_system.send_alert(
                    patient_id=patient_id,
                    alert_type="Contraindication",
                    message=contra['message'],
                    severity="High"
                )

# ===============================
# 7. STREAMLIT DASHBOARD COMPONENTS
# ===============================

class StreamlitDashboard:
    def __init__(self, analyzer, patient_manager):
        self.analyzer = analyzer
        self.patient_manager = patient_manager
    
    def display_patient_dashboard(self, patient_id):
        with st.spinner(f"Analyzing patient {patient_id}..."):
            analysis = self.analyzer.comprehensive_patient_analysis(patient_id)
        
        st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>🏥 Patient Dashboard - {patient_id}</h1>", unsafe_allow_html=True)
        
        # Create tabs for different analysis sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "🧠 AI Knowledge Analysis", 
            "💊 Medications", 
            "🚨 Alerts & Monitoring",
            "📈 Next Best Actions"
        ])
        
        with tab1:
            self._display_patient_overview(analysis['patient_info'])
            self._display_adr_prediction(analysis['adr_prediction'])
            
        with tab2:
            self._display_knowledge_analysis(analysis['knowledge_analysis'], patient_id)
            
        with tab3:
            self._display_medication_analysis(analysis['prescriptions'])
            self._display_real_interactions(analysis['real_database_interactions'])
        
        with tab4:
            self._display_active_alerts()
            self._display_monitoring_section(patient_id)
            
        with tab5:
            self._display_next_best_actions(analysis['next_best_actions'])
        
        return analysis
    
    def _display_patient_overview(self, patient_info):
        st.subheader("👤 Patient Overview")
        
        # Calculate BMI if weight and height are available
        bmi_info = ""
        if patient_info.get('weight_kg') and patient_info.get('height_cm'):
            height_m = patient_info['height_cm'] / 100
            bmi = patient_info['weight_kg'] / (height_m ** 2)
            bmi_category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
            bmi_info = f"<p><strong>BMI:</strong> {bmi:.1f} ({bmi_category})</p>"
        
        with st.container():
            st.markdown(f"""
            <div class='metric-card'>
                <h4>Patient Information</h4>
                <p><strong>ID:</strong> {patient_info['patient_id']}</p>
                <p><strong>Name:</strong> {patient_info.get('first_name', 'N/A')} {patient_info.get('last_name', 'N/A')}</p>
                <p><strong>Age:</strong> {patient_info['age']} | <strong>Gender:</strong> {patient_info.get('gender', 'N/A')}</p>
                <p><strong>Blood Type:</strong> {patient_info.get('blood_type', 'N/A')}</p>
                {bmi_info}
                <p><strong>Conditions:</strong> {', '.join(patient_info['medical_conditions'])}</p>
                <p><strong>Allergies:</strong> {', '.join(patient_info['allergies'])}</p>
                <p><strong>Kidney Function:</strong> {patient_info.get('kidney_function', 'Normal')}</p>
                <p><strong>Liver Function:</strong> {patient_info.get('liver_function', 'Normal')}</p>
                <p><strong>Primary Physician:</strong> {patient_info.get('primary_physician', 'N/A')}</p>
                <p><strong>Room:</strong> {patient_info.get('room_number', 'N/A')}</p>
                <p><strong>Emergency Contact:</strong> {patient_info.get('emergency_contact', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def _display_adr_prediction(self, adr_prediction):
        st.subheader("🤖 ADR Risk Prediction")
        
        risk_color = {
            'Low': 'green',
            'Medium': 'orange', 
            'High': 'red'
        }
        
        with st.container():
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = adr_prediction['probability'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"ADR Risk: {adr_prediction['risk_level']}"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': risk_color[adr_prediction['risk_level']]},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Risk Factors:**")
            for factor in adr_prediction['factors_contributing']:
                st.write(f"• {factor}")
    
    def _display_knowledge_analysis(self, knowledge_analysis, patient_id):
        st.subheader("🧠 Knowledge-Based AI Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk Assessment
            st.markdown("### 📊 Risk Assessment")
            risk = knowledge_analysis['risk_assessment']
            
            # Use custom CSS classes for risk level coloring
            risk_class = {
                'High': 'risk-high',
                'Medium': 'risk-medium',
                'Low': 'risk-low'
            }
            
            st.metric("Overall Risk Score", f"{risk['risk_score']:.2f}")
            
            # Display risk level with colored text
            risk_level_class = risk_class.get(risk['risk_level'], '')
            st.markdown(f"<p class='{risk_level_class}'>Risk Level: {risk['risk_level']}</p>", unsafe_allow_html=True)
            
            st.metric("Number of Risk Factors", risk['num_risk_factors'])
            
            st.write("**Identified Risk Factors:**")
            for factor in risk['risk_factors']:
                st.write(f"• {factor}")
        
        with col2:
            # Drug Suitability
            st.markdown("### 💊 Drug Suitability Analysis")
            suitability = knowledge_analysis['drug_suitability']
            
            st.metric("Suitability Issues", suitability['total_issues'])
            
            if suitability['recommendations']:
                st.success("**Appropriate Medications:**")
                for rec in suitability['recommendations']:
                    st.write(f"✓ {rec}")
            
            if suitability['suitability_issues']:
                st.error("**Medication Issues:**")
                for issue in suitability['suitability_issues']:
                    st.write(f"⚠️ **{issue['drug_name']}**: {', '.join(issue['issues'])}")
        
        # Care Gaps
        st.markdown("### 🔍 Identified Care Gaps")
        care_gaps = knowledge_analysis['care_gaps']
        
        if care_gaps:
            for gap in care_gaps:
                if gap['type'] == 'untreated_condition':
                    st.warning(f"**Untreated Condition**: {gap['message']}")
                    st.write(f"Suggested treatments: {', '.join(gap['suggested_treatments'])}")
                elif gap['type'] == 'monitoring_required':
                    st.info(f"**Monitoring Required**: {gap['message']}")
                    st.write(f"Tests needed: {', '.join(gap['monitoring_tests'])}")
                elif gap['type'] == 'geriatric_assessment':
                    st.info(f"**Geriatric Care**: {gap['message']}")
                    st.write(f"Assessments: {', '.join(gap['suggested_actions'])}")
        else:
            st.success("✅ No significant care gaps identified")
        
        # Knowledge Insights
        st.markdown("### 💡 Clinical Insights")
        insights = knowledge_analysis['knowledge_insights']
        
        if insights:
            for insight in insights:
                if insight['severity'] == 'High':
                    st.error(f"**{insight['type']}**: {insight['message']}")
                    st.write(f"Detail: {insight['detail']}")
                else:
                    st.warning(f"**{insight['type']}**: {insight['message']}")
                    st.write(f"Detail: {insight['detail']}")
        else:
            st.info("No additional clinical insights at this time")
        
        # Semantic Network Visualization
        st.markdown("### 🕸️ Patient Semantic Network")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("Generate Enhanced Semantic Network"):
                with st.spinner("Building detailed patient semantic network..."):
                    fig = self.analyzer.knowledge_agent.visualize_semantic_net(patient_id)
                    st.pyplot(fig)
                    st.caption("Enhanced semantic network showing feature categories and relationships")
        
        with col2:
            st.markdown("""
            **Feature Categories:**
            - 🔴 **Patient**: Central node with demographics
            - 🟢 **Conditions**: Medical diagnoses
            - 🔵 **Drugs**: Current and potential medications
            - 🟠 **Allergies**: Patient allergies
            - 🟣 **Organs**: Organ function status
            - ⚫ **Relationships**: Clinical connections
            """)
    
    def _display_medication_analysis(self, prescriptions):
        st.subheader("💊 Medication Analysis")
        
        for i, drug in enumerate(prescriptions):
            with st.expander(f"{drug['drug_name']} - {drug['dosage']}", expanded=True):
                
                if drug['contraindications']:
                    st.error("**Contraindications:**")
                    for contra in drug['contraindications']:
                        st.write(f"• {contra['message']}")
                
                if drug['fda_interactions']:
                    st.warning("**FDA Information:**")
                    st.write(f"• {drug['fda_interactions'][0]}")
                
                if drug['monitoring_requirements']:
                    st.info("**Monitoring Required:**")
                    for req in drug['monitoring_requirements']:
                        st.write(f"• {req}")
    
    def _display_real_interactions(self, interactions):
        st.subheader("🔗 Drug Interactions")
        
        if interactions:
            for interaction in interactions:
                if interaction['risk_level'] == 'High':
                    st.error(f"**{interaction['drug1']} + {interaction['drug2']}**")
                    st.write(f"• {interaction['message']}")
                    st.write(f"• Source: {interaction['source']}")
                else:
                    st.warning(f"**{interaction['drug1']} + {interaction['drug2']}**")
                    st.write(f"• {interaction['message']}")
        else:
            st.success("✅ No significant drug interactions detected")
    
    def _display_active_alerts(self):
        st.subheader("🚨 Active Alerts")
        
        pending_alerts = self.analyzer.alert_system.get_pending_alerts()
        
        if pending_alerts:
            for i, alert in enumerate(pending_alerts):
                if alert['severity'] == 'High':
                    st.markdown(f"""
                    <div class='alert-high'>
                        <strong>{alert['alert_type']}</strong><br>
                        {alert['message']}<br>
                        <small>Patient: {alert['patient_id']} | {alert['timestamp'].strftime('%Y-%m-%d %H:%M')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                elif alert['severity'] == 'Medium':
                    st.markdown(f"""
                    <div class='alert-medium'>
                        <strong>{alert['alert_type']}</strong><br>
                        {alert['message']}<br>
                        <small>Patient: {alert['patient_id']} | {alert['timestamp'].strftime('%Y-%m-%d %H:%M')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='alert-low'>
                        <strong>{alert['alert_type']}</strong><br>
                        {alert['message']}<br>
                        <small>Patient: {alert['patient_id']} | {alert['timestamp'].strftime('%Y-%m-%d %H:%M')}</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("✅ No active alerts")
    
    def _display_monitoring_section(self, patient_id):
        st.subheader("📊 Real-time Monitoring")
        
        vitals = self.analyzer.monitor.simulate_vital_signs_update(patient_id)
        
        if vitals:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig_hr = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = vitals['heart_rate'],
                    title = {'text': "Heart Rate"},
                    gauge = {
                        'axis': {'range': [40, 160]},
                        'bar': {'color': "red"},
                        'steps': [
                            {'range': [40, 60], 'color': "lightgray"},
                            {'range': [60, 100], 'color': "lightgreen"},
                            {'range': [100, 160], 'color': "lightcoral"}
                        ]
                    }
                ))
                fig_hr.update_layout(height=200)
                st.plotly_chart(fig_hr, use_container_width=True)
            
            with col2:
                st.metric("Blood Pressure", 
                         f"{vitals['blood_pressure_systolic']}/{vitals['blood_pressure_diastolic']} mmHg")
                st.metric("Temperature", f"{vitals['temperature']}°C")
            
            with col3:
                st.metric("Respiratory Rate", f"{vitals['respiratory_rate']} breaths/min")
                st.metric("O₂ Saturation", f"{vitals['oxygen_saturation']}%")
    
    def _display_next_best_actions(self, next_best_actions):
        st.subheader("🎯 Next Best Actions")
        
        st.info(f"**Executive Summary**: {next_best_actions['summary']}")
        
        st.metric("Total Recommended Actions", next_best_actions['total_actions'])
        st.metric("High Priority Actions", next_best_actions['high_priority_actions'])
        
        actions = next_best_actions['actions']
        
        if actions:
            for i, action in enumerate(actions):
                if action['priority'] == 'High':
                    st.markdown(f"""
                    <div class='alert-high'>
                        <h4>🚨 {action['action']}</h4>
                        <p><strong>Priority:</strong> {action['priority']} | <strong>Timeline:</strong> {action['timeline']}</p>
                        <p><strong>Reason:</strong> {action['reason']}</p>
                        <p><strong>Responsible:</strong> {action['responsible']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif action['priority'] == 'Medium':
                    st.markdown(f"""
                    <div class='alert-medium'>
                        <h4>⚠️ {action['action']}</h4>
                        <p><strong>Priority:</strong> {action['priority']} | <strong>Timeline:</strong> {action['timeline']}</p>
                        <p><strong>Reason:</strong> {action['reason']}</p>
                        <p><strong>Responsible:</strong> {action['responsible']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='alert-low'>
                        <h4>ℹ️ {action['action']}</h4>
                        <p><strong>Priority:</strong> {action['priority']} | <strong>Timeline:</strong> {action['timeline']}</p>
                        <p><strong>Reason:</strong> {action['reason']}</p>
                        <p><strong>Responsible:</strong> {action['responsible']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("✅ No specific actions recommended. Current management appears appropriate.")

# ===============================
# 8. MAIN STREAMLIT APP
# ===============================

def main():
    st.markdown("<h1 class='main-header'>🏥 Drug Suitability Analysis Platform</h1>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'patient_manager' not in st.session_state:
        st.session_state.patient_manager = PatientDataManager()
        
        # Try to load existing data, otherwise generate sample data
        loaded_patient_df, loaded_prescription_df = st.session_state.patient_manager.load_data()
        
        if loaded_patient_df is not None and loaded_prescription_df is not None:
            st.session_state.patient_df = loaded_patient_df
            st.session_state.prescription_df = loaded_prescription_df
            st.sidebar.success("✅ Loaded existing patient data")
        else:
            patient_df, prescription_df = st.session_state.patient_manager.generate_large_dataset()
            st.session_state.patient_df = patient_df
            st.session_state.prescription_df = prescription_df
            st.session_state.patient_manager.save_data(patient_df, prescription_df)
            st.sidebar.info("📊 Generated new sample data")
        
        st.session_state.analyzer = EnhancedDrugAnalyzer(
            st.session_state.patient_df, st.session_state.prescription_df
        )
        st.session_state.dashboard = StreamlitDashboard(
            st.session_state.analyzer, st.session_state.patient_manager
        )
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", [
        "Patient Dashboard", 
        "Patient Management", 
        "Add New Patient", 
        "Edit Patient",
        "Delete Patient",
        "Data Management",
        "Population Analytics"
    ])
    
    # System overview in sidebar
    st.sidebar.title("System Overview")
    st.sidebar.metric("Total Patients", len(st.session_state.patient_df))
    st.sidebar.metric("Active Alerts", len(st.session_state.analyzer.alert_system.get_pending_alerts()))
    
    # Features info
    st.sidebar.title("Enhanced Features")
    st.sidebar.info("""
    - **🧠 Knowledge-Based AI Agent**
    - **🤖 AI-Powered ADR Prediction**
    - **💾 Persistent Data Storage**
    - **👥 Patient CRUD Operations**
    - **🔗 Real Database Integration**
    - **📊 Real-time Monitoring**
    - **💊 Drug Interaction Detection**
    - **🚨 Automated Alert System**
    - **🎯 Next Best Actions**
    - **📈 Population Analytics**
    """)
    
    # Main content based on selected page
    if page == "Patient Dashboard":
        display_patient_dashboard()
    elif page == "Patient Management":
        display_patient_management()
    elif page == "Add New Patient":
        display_add_patient()
    elif page == "Edit Patient":
        display_edit_patient()
    elif page == "Delete Patient":
        display_delete_patient()
    elif page == "Data Management":
        display_data_management()
    elif page == "Population Analytics":
        display_population_analytics()
    
    # Save data automatically when changes are made
    if 'patient_df' in st.session_state and 'prescription_df' in st.session_state:
        st.session_state.patient_manager.save_data(
            st.session_state.patient_df, st.session_state.prescription_df
        )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh System"):
        st.rerun()
    
    # System summary at the bottom
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    st.sidebar.success("✅ All systems operational")
    st.sidebar.info("🔄 Real-time monitoring active")
    st.sidebar.success("🧠 Knowledge AI Agent active")
    st.sidebar.success("💾 Data persistence active")

def display_patient_dashboard():
    """Display the main patient dashboard"""
    st.sidebar.subheader("Patient Selection")
    selected_patient = st.sidebar.selectbox(
        "Choose a patient:",
        st.session_state.patient_df['patient_id'].tolist()
    )
    
    if selected_patient:
        # Get patient name for display
        patient_data = st.session_state.patient_df[st.session_state.patient_df['patient_id'] == selected_patient].iloc[0]
        patient_name = f"{patient_data.get('first_name', '')} {patient_data.get('last_name', '')}"
        
        st.info(f"**Currently viewing:** {patient_name} ({selected_patient})")
        st.session_state.dashboard.display_patient_dashboard(selected_patient)

def display_patient_management():
    """Display patient management interface"""
    st.header("📋 Patient Management")
    
    # Quick statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='stats-card'>
            <h3>{len(st.session_state.patient_df)}</h3>
            <p>Total Patients</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_age = st.session_state.patient_df['age'].mean()
        st.markdown(f"""
        <div class='stats-card'>
            <h3>{avg_age:.1f}</h3>
            <p>Average Age</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        female_count = len(st.session_state.patient_df[st.session_state.patient_df['gender'] == 'Female'])
        st.markdown(f"""
        <div class='stats-card'>
            <h3>{female_count}</h3>
            <p>Female Patients</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        male_count = len(st.session_state.patient_df[st.session_state.patient_df['gender'] == 'Male'])
        st.markdown(f"""
        <div class='stats-card'>
            <h3>{male_count}</h3>
            <p>Male Patients</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display all patients in a table
    st.subheader("Current Patients")
    
    if not st.session_state.patient_df.empty:
        # Create a display dataframe with selected columns
        display_df = st.session_state.patient_df[[
            'patient_id', 'first_name', 'last_name', 'age', 'gender', 
            'primary_physician', 'room_number', 'admission_date'
        ]].copy()
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Patient details
        st.subheader("Patient Details")
        selected_patient = st.selectbox(
            "Select patient for details:",
            st.session_state.patient_df['patient_id'].tolist()
        )
        
        if selected_patient:
            patient_data = st.session_state.patient_df[st.session_state.patient_df['patient_id'] == selected_patient].iloc[0]
            prescriptions = st.session_state.prescription_df[st.session_state.prescription_df['patient_id'] == selected_patient]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Personal Information:**")
                st.write(f"**Name:** {patient_data.get('first_name', '')} {patient_data.get('last_name', '')}")
                st.write(f"**Age:** {patient_data['age']}")
                st.write(f"**Gender:** {patient_data.get('gender', 'N/A')}")
                st.write(f"**Blood Type:** {patient_data.get('blood_type', 'N/A')}")
                st.write(f"**Weight:** {patient_data.get('weight_kg', 'N/A')} kg")
                st.write(f"**Height:** {patient_data.get('height_cm', 'N/A')} cm")
                
                if patient_data.get('weight_kg') and patient_data.get('height_cm'):
                    height_m = patient_data['height_cm'] / 100
                    bmi = patient_data['weight_kg'] / (height_m ** 2)
                    bmi_category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
                    st.write(f"**BMI:** {bmi:.1f} ({bmi_category})")
            
            with col2:
                st.write("**Medical Information:**")
                st.write(f"**Conditions:** {', '.join(patient_data['medical_conditions'])}")
                st.write(f"**Allergies:** {', '.join(patient_data['allergies'])}")
                st.write(f"**Kidney Function:** {patient_data.get('kidney_function', 'N/A')}")
                st.write(f"**Liver Function:** {patient_data.get('liver_function', 'N/A')}")
                st.write(f"**Comorbidity Score:** {patient_data.get('comorbidity_score', 'N/A')}")
                st.write(f"**Previous ADR:** {'Yes' if patient_data.get('previous_adr_history', 0) == 1 else 'No'}")
                st.write(f"**Primary Physician:** {patient_data.get('primary_physician', 'N/A')}")
                st.write(f"**Room:** {patient_data.get('room_number', 'N/A')}")
                st.write(f"**Emergency Contact:** {patient_data.get('emergency_contact', 'N/A')}")
            
            # Display prescriptions
            st.subheader("Current Prescriptions")
            if not prescriptions.empty:
                st.dataframe(prescriptions, use_container_width=True)
            else:
                st.info("No prescriptions found for this patient.")
            
            # Add prescription for this patient
            st.subheader("Add New Prescription")
            st.session_state.prescription_df = st.session_state.patient_manager.add_prescription(
                selected_patient, st.session_state.prescription_df
            )
    else:
        st.info("No patients in the system. Please add a new patient.")

def display_add_patient():
    """Display interface for adding new patients"""
    st.header("➕ Add New Patient")
    
    st.session_state.patient_df, st.session_state.prescription_df = st.session_state.patient_manager.add_new_patient(
        st.session_state.patient_df, st.session_state.prescription_df
    )
    
    # Update analyzer with new data
    if st.session_state.patient_df is not None:
        st.session_state.analyzer = EnhancedDrugAnalyzer(st.session_state.patient_df, st.session_state.prescription_df)
        st.session_state.dashboard = StreamlitDashboard(st.session_state.analyzer, st.session_state.patient_manager)

def display_edit_patient():
    """Display interface for editing patients"""
    st.header("✏️ Edit Patient")
    
    if st.session_state.patient_df.empty:
        st.info("No patients in the system. Please add a new patient first.")
        return
    
    selected_patient = st.selectbox(
        "Select patient to edit:",
        st.session_state.patient_df['patient_id'].tolist()
    )
    
    if selected_patient:
        st.session_state.patient_df = st.session_state.patient_manager.edit_patient(
            selected_patient, st.session_state.patient_df
        )

def display_delete_patient():
    """Display interface for deleting patients"""
    st.header("🗑️ Delete Patient")
    
    if st.session_state.patient_df.empty:
        st.info("No patients in the system. Please add a new patient first.")
        return
    
    selected_patient = st.selectbox(
        "Select patient to delete:",
        st.session_state.patient_df['patient_id'].tolist()
    )
    
    if selected_patient:
        # Show patient details before deletion
        patient_data = st.session_state.patient_df[st.session_state.patient_df['patient_id'] == selected_patient].iloc[0]
        patient_name = f"{patient_data.get('first_name', '')} {patient_data.get('last_name', '')}"
        
        st.warning(f"### You are about to delete patient: {patient_name} ({selected_patient})")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Patient Details:**")
            st.write(f"**Age:** {patient_data['age']}")
            st.write(f"**Gender:** {patient_data.get('gender', 'N/A')}")
            st.write(f"**Conditions:** {', '.join(patient_data['medical_conditions'])}")
            st.write(f"**Allergies:** {', '.join(patient_data['allergies'])}")
        
        with col2:
            # Show prescriptions that will also be deleted
            prescriptions = st.session_state.prescription_df[
                st.session_state.prescription_df['patient_id'] == selected_patient
            ]
            st.write("**Prescriptions to be deleted:**")
            if not prescriptions.empty:
                for _, prescription in prescriptions.iterrows():
                    st.write(f"- {prescription['drug_name']} ({prescription['dosage']})")
            else:
                st.write("No prescriptions found for this patient.")
        
        # Confirmation
        st.error("⚠️ **This action cannot be undone!**")
        
        if st.button("🚨 Confirm Deletion", type="primary"):
            st.session_state.patient_df, st.session_state.prescription_df = st.session_state.patient_manager.delete_patient(
                selected_patient, st.session_state.patient_df, st.session_state.prescription_df
            )
            
            # Update analyzer with new data
            st.session_state.analyzer = EnhancedDrugAnalyzer(st.session_state.patient_df, st.session_state.prescription_df)
            st.session_state.dashboard = StreamlitDashboard(st.session_state.analyzer, st.session_state.patient_manager)
            
            st.success(f"Patient {selected_patient} has been deleted successfully!")
            st.rerun()

def display_data_management():
    """Display data management interface"""
    st.header("💾 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Statistics")
        st.metric("Total Patients", len(st.session_state.patient_df))
        st.metric("Total Prescriptions", len(st.session_state.prescription_df))
        st.metric("Data Directory", "patient_data/")
        
        # Backup functionality
        st.subheader("Backup & Restore")
        if st.button("📂 Create Data Backup"):
            if st.session_state.patient_manager.storage_manager.backup_data():
                st.success("Data backup created successfully!")
            else:
                st.error("Failed to create data backup")
    
    with col2:
        st.subheader("Data Operations")
        
        # Export data
        if st.button("📤 Export Patient Data"):
            csv_data = st.session_state.patient_df.to_csv(index=False)
            st.download_button(
                label="Download Patient CSV",
                data=csv_data,
                file_name="patients_export.csv",
                mime="text/csv"
            )
        
        if st.button("📤 Export Prescription Data"):
            csv_data = st.session_state.prescription_df.to_csv(index=False)
            st.download_button(
                label="Download Prescription CSV",
                data=csv_data,
                file_name="prescriptions_export.csv",
                mime="text/csv"
            )
        
        # Reset data
        st.subheader("Reset Data")
        st.warning("This will delete all current data and generate new sample data.")
        if st.button("🔄 Reset to Sample Data"):
            patient_df, prescription_df = st.session_state.patient_manager.generate_large_dataset()
            st.session_state.patient_df = patient_df
            st.session_state.prescription_df = prescription_df
            st.session_state.patient_manager.save_data(patient_df, prescription_df)
            st.session_state.analyzer = EnhancedDrugAnalyzer(patient_df, prescription_df)
            st.session_state.dashboard = StreamlitDashboard(st.session_state.analyzer, st.session_state.patient_manager)
            st.success("Data reset successfully!")
            st.rerun()

def display_population_analytics():
    """Display population-level analytics"""
    st.header("📊 Population Analytics")
    
    # Get statistics
    stats = st.session_state.patient_manager.get_patient_statistics(st.session_state.patient_df)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", stats['total_patients'])
    
    with col2:
        st.metric("Average Age", f"{stats['avg_age']:.1f}")
    
    with col3:
        st.metric("Female Patients", stats['gender_distribution'].get('Female', 0))
    
    with col4:
        st.metric("Male Patients", stats['gender_distribution'].get('Male', 0))
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        st.subheader("Age Distribution")
        fig_age = px.histogram(st.session_state.patient_df, x='age', nbins=20, 
                              title="Patient Age Distribution")
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Gender distribution
        st.subheader("Gender Distribution")
        gender_counts = pd.Series(stats['gender_distribution'])
        fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index,
                           title="Patient Gender Distribution")
        st.plotly_chart(fig_gender, use_container_width=True)
    
    # Medical conditions
    st.subheader("Top Medical Conditions")
    conditions_df = pd.DataFrame(stats['common_conditions'], columns=['Condition', 'Count'])
    fig_conditions = px.bar(conditions_df, x='Condition', y='Count',
                           title="Most Common Medical Conditions")
    st.plotly_chart(fig_conditions, use_container_width=True)
    
    # Comorbidity scores
    st.subheader("Comorbidity Score Distribution")
    fig_comorbidity = px.histogram(st.session_state.patient_df, x='comorbidity_score',
                                  title="Comorbidity Score Distribution")
    st.plotly_chart(fig_comorbidity, use_container_width=True)
    
    # Age groups
    st.subheader("Patients by Age Group")
    age_groups_df = pd.DataFrame(list(stats['age_groups'].items()), 
                                columns=['Age Group', 'Count'])
    fig_age_groups = px.bar(age_groups_df, x='Age Group', y='Count',
                           title="Patients by Age Group")
    st.plotly_chart(fig_age_groups, use_container_width=True)

if __name__ == "__main__":
    main()