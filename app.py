
import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Student Performance & Learning Insights",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# LOAD DATA AND MODEL
# ============================================================

df = pd.read_csv("student_data.csv")
model = joblib.load("student_performance_model.pkl")

features = [
    "study_hours",
    "attendance",
    "assignment_score",
    "previous_score",
    "maths",
    "dbms",
    "python",
    "os",
    "ml"
]

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 800;
    text-align: center;
    margin-top: 5px;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    opacity: 0.70;
    margin-bottom: 30px;
}

.section-title {
    font-size: 25px;
    font-weight: 750;
    margin-top: 32px;
    margin-bottom: 18px;
}

.card {
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.30);
    background: rgba(128,128,128,0.08);
    text-align: center;
    min-height: 120px;
}

.card-title {
    font-size: 15px;
    opacity: 0.70;
    margin-bottom: 8px;
}

.card-value {
    font-size: 26px;
    font-weight: 750;
}

.prediction-card {
    padding: 24px;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,0.30);
    background: rgba(128,128,128,0.08);
    text-align: center;
    min-height: 150px;
}

.prediction-label {
    font-size: 16px;
    opacity: 0.70;
    margin-bottom: 12px;
}

.prediction-value {
    font-size: 38px;
    font-weight: 800;
}

.info-box {
    padding: 20px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.30);
    background: rgba(128,128,128,0.08);
    line-height: 1.7;
}

.action-item {
    padding: 14px 18px;
    margin: 9px 0;
    border-radius: 10px;
    border: 1px solid rgba(128,128,128,0.25);
    background: rgba(128,128,128,0.06);
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 AI Student Performance & Learning Insights</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered academic performance analysis and personalized learning recommendations'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# STUDENT SEARCH
# ============================================================

st.markdown(
    '<div class="section-title">🔎 Student Report</div>',
    unsafe_allow_html=True
)

student_id = st.text_input(
    "Enter Student ID",
    placeholder="Example: ST0007"
)

generate = st.button(
    "🚀 Generate Student Report",
    use_container_width=True
)

# ============================================================
# GENERATE REPORT
# ============================================================

if generate:

    student_id = student_id.strip().upper()

    if student_id == "":
        st.warning("Please enter a Student ID.")

    elif student_id not in df["student_id"].values:

        st.error(
            "❌ Student ID not found. Please check the ID and try again."
        )

    else:

        # ====================================================
        # GET STUDENT
        # ====================================================

        student = df[df["student_id"] == student_id].iloc[0]

        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        X_student = pd.DataFrame(
            [[student[feature] for feature in features]],
            columns=features
        )

        prediction = model.predict(X_student)[0]

        # ====================================================
        # REPORT HEADER
        # ====================================================

        st.markdown(
            f"### 📋 Performance Report — {student_id}"
        )

        # ====================================================
        # SUMMARY CARDS
        # ====================================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">Current Score</div>
                    <div class="card-value">
                        {student["current_score"]:.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">Performance</div>
                    <div class="card-value">
                        {student["performance_level"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">Strongest Subject</div>
                    <div class="card-value">
                        {str(student["strongest_subject"]).upper()}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col4:

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">Weakest Subject</div>
                    <div class="card-value">
                        {str(student["weakest_subject"]).upper()}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ====================================================
        # AI PERFORMANCE PREDICTION
        # ====================================================

        st.markdown("### 🤖 AI Performance Prediction")

        pred_col1, pred_col2 = st.columns(2)

        with pred_col1:
            st.metric(
                label="🎯 Predicted Score",
                value=f"{prediction:.2f}",
                help="Score predicted by the trained Random Forest model"
            )

        with pred_col2:
            st.metric(
                label="📊 Predicted Performance",
                value=str(student["predicted_performance"]),
                help="Performance category based on the predicted score"
                    )
        # ====================================================
        # CURRENT VS PREDICTED SCORE
        # ====================================================

        st.markdown("### 📈 Current vs Predicted Score")

        comparison_data = pd.DataFrame(
            {
                "Score": [
                    float(student["current_score"]),
                    float(prediction)
                ]
            },
            index=[
                "Current Score",
                "Predicted Future Score"
            ]
        )

        st.bar_chart(
            comparison_data,
            use_container_width=True
        )

        difference = float(prediction) - float(student["current_score"])

        if difference > 0:
            st.success(
                f"📈 The model predicts an improvement of {difference:.2f} points."
            )
        elif difference < 0:
            st.warning(
                f"📉 The model predicts a decrease of {abs(difference):.2f} points."
            )
        else:
            st.info(
                "➡️ The model predicts a similar future score."
            )

# ====================================================
# SUBJECT-WISE PERFORMANCE
# ====================================================

        st.markdown(
            "### 📚 Subject-wise Performance"
        )

        subject_data = pd.DataFrame(
            {
                "Subject": [
                    "Maths",
                    "DBMS",
                    "Python",
                    "OS",
                    "ML"
                ],
                "Score": [
                    int(student["maths"]),
                    int(student["dbms"]),
                    int(student["python"]),
                    int(student["os"]),
                    int(student["ml"])
                ],
                "Status": [
                    "Needs Improvement" if student["maths"] < 50
                    else "Average" if student["maths"] < 70
                    else "Good",

                    "Needs Improvement" if student["dbms"] < 50
                    else "Average" if student["dbms"] < 70
                    else "Good",

                    "Needs Improvement" if student["python"] < 50
                    else "Average" if student["python"] < 70
                    else "Good",

                    "Needs Improvement" if student["os"] < 50
                    else "Average" if student["os"] < 70
                    else "Good",

                    "Needs Improvement" if student["ml"] < 50
                    else "Average" if student["ml"] < 70
                    else "Good"
                ]
            }
        )

        # Subject table
        st.dataframe(
        subject_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Subject": st.column_config.TextColumn(
                "📚 Subject"
            ),
            "Score": st.column_config.NumberColumn(
                "🎯 Score",
                format="%.0f"
            ),
            "Status": st.column_config.TextColumn(
                "📊 Status"
            )
        }
    )

        # Subject graph
        st.bar_chart(
            subject_data.set_index("Subject")["Score"]
        )

        # ====================================================
        # STUDY PROFILE
        # ====================================================

        st.markdown(
            "### 📊 Study Profile"
        )

        profile_col1, profile_col2, profile_col3 = st.columns(3)

        with profile_col1:

            st.metric(
                "⏱️ Study Hours",
                f'{student["study_hours"]} hrs/week'
            )

        with profile_col2:

            st.metric(
                "📅 Attendance",
                f'{student["attendance"]}%'
            )

        with profile_col3:

            st.metric(
                "📝 Assignment Score",
                f'{student["assignment_score"]}'
            )

        # ====================================================
        # AI PERFORMANCE SUMMARY
        # ====================================================

        st.markdown("### 📝 AI Performance Summary")

        summary = (
            f"{student_id} is currently performing at a "
            f"{student['performance_level']} level with a current score of "
            f"{student['current_score']:.2f}. "
            f"{str(student['strongest_subject']).upper()} is the strongest subject, "
            f"while {str(student['weakest_subject']).upper()} requires the most attention. "
            f"The AI model predicts a future score of {prediction:.2f}, "
            f"with a predicted performance level of "
            f"{student['predicted_performance']}."
        )

        st.info(summary)
        # ====================================================
        # PERSONALIZED LEARNING INSIGHT
        # ====================================================

        st.markdown(
            "### 🧠 Personalized Learning Insight"
        )

        st.info(
            str(student["detailed_insight"])
        )

        # ====================================================
        # LEARNING RECOMMENDATION
        # ====================================================

        st.markdown(
            "### 💡 Learning Recommendation"
        )

        st.success(
            str(student["learning_recommendation"])
        )

        # ====================================================
        # RECOMMENDED ACTION PLAN
        # ====================================================

        st.markdown(
            "### ✅ Recommended Action Plan"
        )

        weakest = str(
            student["weakest_subject"]
        ).upper()

        st.markdown(
            f"🎯 **Focus:** Give additional attention to **{weakest}** during your study sessions."
        )

        st.markdown(
            "📚 **Consistency:** Maintain regular study habits and review difficult topics frequently."
        )

        st.markdown(
            "📈 **Improvement:** Strengthen weak areas while maintaining your strongest subjects."
        )
        # ====================================================
        # DOWNLOAD REPORT
        # ====================================================

        st.markdown("### 📥 Download Student Report")

        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle
        )
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.units import mm
        from io import BytesIO


        # ------------------------------------------------------------
        # CREATE PDF
        # ------------------------------------------------------------

        pdf_buffer = BytesIO()

        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm
        )

        styles = getSampleStyleSheet()

        title_style = styles["Title"]
        title_style.alignment = TA_CENTER

        heading_style = styles["Heading2"]

        normal_style = styles["BodyText"]


        story = []


        # ------------------------------------------------------------
        # TITLE
        # ------------------------------------------------------------

        story.append(
            Paragraph(
                "AI Student Performance & Learning Insights",
                title_style
            )
        )

        story.append(
            Spacer(1, 8)
        )

        story.append(
            Paragraph(
                f"<b>Student ID:</b> {student_id}",
                normal_style
            )
        )

        story.append(
            Spacer(1, 15)
        )


        # ------------------------------------------------------------
        # PERFORMANCE OVERVIEW
        # ------------------------------------------------------------

        story.append(
            Paragraph(
                "Performance Overview",
                heading_style
            )
        )

        overview_data = [
            ["Current Score", "Performance", "Strongest Subject", "Weakest Subject"],
            [
                f'{student["current_score"]:.2f}',
                str(student["performance_level"]),
                str(student["strongest_subject"]).upper(),
                str(student["weakest_subject"]).upper()
            ]
        ]

        overview_table = Table(
            overview_data,
            colWidths=[40 * mm, 40 * mm, 45 * mm, 45 * mm]
        )

        overview_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ])
        )

        story.append(overview_table)

        story.append(
            Spacer(1, 18)
        )


        # ------------------------------------------------------------
        # AI PREDICTION
        # ------------------------------------------------------------

        story.append(
            Paragraph(
                "AI Performance Prediction",
                heading_style
            )
        )

        prediction_data = [
            ["Predicted Future Score", "Predicted Performance"],
            [
                f"{prediction:.2f}",
                str(student["predicted_performance"])
            ]
        ]

        prediction_table = Table(
            prediction_data,
            colWidths=[90 * mm, 90 * mm]
        )

        prediction_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 1), (-1, 1), 16),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ])
        )

        story.append(prediction_table)

        story.append(
            Spacer(1, 18)
        )


        # ------------------------------------------------------------
        # SUBJECT-WISE PERFORMANCE
        # ------------------------------------------------------------

        story.append(
            Paragraph(
                "Subject-wise Performance",
                heading_style
            )
        )

        subject_pdf_data = [
            ["Subject", "Score", "Status"],
            [
                "Maths",
                int(student["maths"]),
                "Needs Improvement"
                if student["maths"] < 50
                else "Average"
                if student["maths"] < 70
                else "Good"
            ],
            [
                "DBMS",
                int(student["dbms"]),
                "Needs Improvement"
                if student["dbms"] < 50
                else "Average"
                if student["dbms"] < 70
                else "Good"
            ],
            [
                "Python",
                int(student["python"]),
                "Needs Improvement"
                if student["python"] < 50
                else "Average"
                if student["python"] < 70
                else "Good"
            ],
            [
                "OS",
                int(student["os"]),
                "Needs Improvement"
                if student["os"] < 50
                else "Average"
                if student["os"] < 70
                else "Good"
            ],
            [
                "ML",
                int(student["ml"]),
                "Needs Improvement"
                if student["ml"] < 50
                else "Average"
                if student["ml"] < 70
                else "Good"
            ]
        ]

        subject_table = Table(
            subject_pdf_data,
            colWidths=[60 * mm, 45 * mm, 65 * mm]
        )

        subject_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ])
        )

        story.append(subject_table)

        story.append(
            Spacer(1, 18)
        )


        # ------------------------------------------------------------
        # STUDY PROFILE
        # ------------------------------------------------------------

        story.append(
            Paragraph(
                "Study Profile",
                heading_style
            )
        )

        profile_data = [
            ["Study Hours", "Attendance", "Assignment Score"],
            [
                f'{student["study_hours"]} hrs/week',
                f'{student["attendance"]}%',
                str(student["assignment_score"])
            ]
        ]

        profile_table = Table(
            profile_data,
            colWidths=[60 * mm, 60 * mm, 60 * mm]
        )

        profile_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ])
        )

        story.append(profile_table)

        story.append(
            Spacer(1, 18)
        )


        # ------------------------------------------------------------
        # INSIGHT
        # ------------------------------------------------------------

        story.append(
            Paragraph(
                "Personalized Learning Insight",
                heading_style
            )
        )

        story.append(
            Paragraph(
                str(student["detailed_insight"]),
                normal_style
            )
        )

        story.append(
            Spacer(1, 15)
        )


        # ------------------------------------------------------------
        # RECOMMENDATION
        # ------------------------------------------------------------

        story.append(
            Paragraph(
                "Learning Recommendation",
                heading_style
            )
        )

        story.append(
            Paragraph(
                str(student["learning_recommendation"]),
                normal_style
            )
        )

        story.append(
            Spacer(1, 15)
        )


        # ------------------------------------------------------------
        # ACTION PLAN
        # ------------------------------------------------------------

        story.append(
            Paragraph(
                "Recommended Action Plan",
                heading_style
            )
        )

        story.append(
            Paragraph(
                f"1. Give additional attention to {str(student['weakest_subject']).upper()}.",
                normal_style
            )
        )

        story.append(
            Spacer(1, 6)
        )

        story.append(
            Paragraph(
                "2. Maintain regular study habits and review difficult topics frequently.",
                normal_style
            )
        )

        story.append(
            Spacer(1, 6)
        )

        story.append(
            Paragraph(
                "3. Strengthen weak areas while maintaining your strongest subjects.",
                normal_style
            )
        )


        # ------------------------------------------------------------
        # BUILD PDF
        # ------------------------------------------------------------

        doc.build(story)

        pdf_buffer.seek(0)

        # ------------------------------------------------------------
        # DOWNLOAD BUTTON
        # ------------------------------------------------------------

        st.download_button(
            label="📄 Download Student Report as PDF",
            data=pdf_buffer,
            file_name=f"{student_id}_student_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        # ====================================================
        # FOOTER
        # ====================================================


        st.markdown("---")

        st.caption(
            "AI-Powered Student Performance & Learning Insights • "
            "Machine Learning based analysis"
        )
