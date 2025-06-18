import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from joblib import load

# --- Page Configuration ---
st.set_page_config(layout="wide")

# --- Data Loading and Preparation (Cached) ---
@st.cache_data
def load_data():
    """
    Loads, caches, and prepares the student dataset.
    This function runs only once and the result is stored in cache.
    """
    data = pd.read_csv("dataset/df_new.csv", delimiter=",")
    category_mapping = {
        33: 'Biofuel Production Technologies', 171: 'Animation and Multimedia Design',
        8014: 'Social Service (evening attendance)', 9003: 'Agronomy',
        9070: 'Communication Design', 9085: 'Veterinary Nursing',
        9119: 'Informatics Engineering', 9130: 'Equinculture',
        9147: 'Management', 9238: 'Social Service', 9254: 'Tourism',
        9500: 'Nursing', 9556: 'Oral Hygiene',
        9670: 'Advertising and Marketing Management', 9773: 'Journalism and Communication',
        9853: 'Basic Education', 9991: 'Management (evening attendance)'
    }
    data['Course_Label'] = data['Course'].replace(category_mapping)
    return data

data = load_data()

# --- Reusable UI and Charting Functions ---
def create_styled_card(content, **kwargs):
    """
    Creates a styled card using HTML for displaying metrics.
    Accepts content and optional CSS properties as keyword arguments for customization.
    """
    # Base CSS styles for the card
    styles = {
        'height': '120px', 'border': '1px solid #444', 'border-radius': '8px',
        'font-size': '20px', 'padding': '15px', 'background-color': '#1E1E1E',
        'text-align': 'center', 'display': 'flex', 'justify-content': 'center',
        'align-items': 'center', 'margin-bottom': '10px', 'color': 'white'
    }
    # Allow overriding or adding styles via kwargs
    styles.update(kwargs)
    
    style_str = '; '.join([f'{key}: {value}' for key, value in styles.items()])
    return f"<div style='{style_str}'>{content}</div>"

def get_plotly_dark_theme():
    """Returns a dictionary for a consistent dark theme for Plotly charts."""
    return {
        'plot_bgcolor': '#111111', 'paper_bgcolor': '#111111',
        'font': {'color': 'white'}, 'xaxis_showline': True, 'yaxis_showline': True,
        'xaxis': {'title_font': {'color': 'white'}, 'tickfont': {'color': 'white'}, 'linecolor': 'white', 'linewidth': 1, 'gridcolor': '#333'},
        'yaxis': {'title_font': {'color': 'white'}, 'tickfont': {'color': 'white'}, 'linecolor': 'white', 'linewidth': 1, 'gridcolor': '#333'},
        'margin': {'l': 40, 'r': 40, 't': 50, 'b': 40},
    }

def create_pie_chart(dataframe, column, title):
    """Generates a styled Pie chart for boolean (0/1) data."""
    if dataframe.empty or column not in dataframe.columns or dataframe[column].isnull().all():
        st.info(f"No data for '{title}'.")
        return

    value_counts = dataframe[column].value_counts()
    
    # Map index 0 to 'No' and 1 to 'Yes' for clear labeling
    names = {0: 'No', 1: 'Yes'}
    labels = [names.get(i, i) for i in value_counts.index]

    fig = px.pie(
        values=value_counts.values,
        names=labels,
        title=title,
        color=labels,
        color_discrete_map={'Yes': '#393939', 'No': 'white'}
    )
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=60, b=10),
        title=dict(x=0.05, font=dict(size=15)),
        legend_title_text='',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Sidebar Navigation ---
add_selectbox = st.sidebar.selectbox(
    "Choose a page",
    ("Dashboard", "Prediction")
)

# --- Dashboard Page ---
if add_selectbox == "Dashboard":
    st.title('Jaya Jaya Institute Student Performance Dashboard')

    # --- Sidebar Filters ---
    st.sidebar.header("Dashboard Filters")
    status_options = {'All Students': 'All', 'Dropout': 0, 'Enrolled': 1, 'Graduated': 2}
    selected_status_label = st.sidebar.selectbox('Filter by Status', list(status_options.keys()))
    selected_status = status_options[selected_status_label]
    
    course_list = ['All'] + sorted(data.Course_Label.unique())
    selected_course = st.sidebar.selectbox('Filter by Course', course_list)

    time_options = {'All': 'All', 'Daytime': 1, 'Evening': 0}
    selected_time_label = st.sidebar.selectbox('Filter by Attendance Time', list(time_options.keys()))
    selected_time = time_options[selected_time_label]
    
    gender_options = {'All': 'All', 'Female': 0, 'Male': 1}
    selected_gender_label = st.sidebar.selectbox('Filter by Gender', list(gender_options.keys()))
    selected_gender = gender_options[selected_gender_label]

    # --- Filtering Logic ---
    filtered_data = data.copy()
    if selected_status != 'All':
        filtered_data = filtered_data[filtered_data['Status'] == selected_status]
    if selected_course != 'All':
        filtered_data = filtered_data[filtered_data['Course_Label'] == selected_course]
    if selected_time != 'All':
        filtered_data = filtered_data[filtered_data['Daytime_evening_attendance'] == selected_time]
    if selected_gender != 'All':
        filtered_data = filtered_data[filtered_data['Gender'] == selected_gender]

    # --- Main Metrics Display ---
    st.subheader("Filtered Overview")
    total_students = len(filtered_data)
    if total_students > 0:
        dropout_count = int(filtered_data['Status_0'].sum())
        enrolled_count = int(filtered_data['Status_1'].sum())
        graduated_count = int(filtered_data['Status_2'].sum())
        # Use the full dataset for overall rate calculation unless a specific status is chosen
        rate_base = len(data) if selected_status == 'All' else total_students
        dropout_rate = f"{(data['Status_0'].sum() / rate_base) * 100:.2f}%"

    else:
        dropout_count = enrolled_count = graduated_count = 0
        dropout_rate = "N/A"

    col_rate, col_total, col_do, col_en, col_gr = st.columns(5)
    with col_rate:
         st.markdown(create_styled_card(f"<b>Overall Dropout Rate</b><br><span style='font-size: 32px;'>{dropout_rate}</span>"), unsafe_allow_html=True)
    with col_total:
        st.markdown(create_styled_card(f"<b>Filtered Students</b><br><span style='font-size: 32px;'>{total_students}</span>"), unsafe_allow_html=True)
    with col_do:
        st.markdown(create_styled_card(f"<b>Dropped Out</b><br><span style='font-size: 32px;'>{dropout_count}</span>"), unsafe_allow_html=True)
    with col_en:
        st.markdown(create_styled_card(f"<b>Enrolled</b><br><span style='font-size: 32px;'>{enrolled_count}</span>"), unsafe_allow_html=True)
    with col_gr:
        st.markdown(create_styled_card(f"<b>Graduated</b><br><span style='font-size: 32px;'>{graduated_count}</span>"), unsafe_allow_html=True)

    st.divider()

    # --- Detailed Charts ---
    if not filtered_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Scholarship Holders by Status')
            scholarship_data = filtered_data.groupby('Status')['Scholarship_holder'].sum().reindex([0, 1, 2], fill_value=0)
            
            fig = go.Figure(go.Bar(
                x=['Dropout', 'Enrolled', 'Graduated'], y=scholarship_data.values,
                text=[f'{v}' for v in scholarship_data.values], textposition='auto',
                marker_color='#393939'
            ))
            fig.update_layout(get_plotly_dark_theme(), xaxis_title="Status", yaxis_title="Number of Students")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader('Average Grade per Semester by Status')
            avg_grades = filtered_data.groupby('Status')[['Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_grade']].mean().reindex([0, 1, 2], fill_value=0)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='1st Sem Grade', x=['Dropout', 'Enrolled', 'Graduated'], y=avg_grades['Curricular_units_1st_sem_grade'],
                text=[f'{v:.2f}' for v in avg_grades['Curricular_units_1st_sem_grade']], textposition='auto', marker_color='#393939'
            ))
            fig.add_trace(go.Bar(
                name='2nd Sem Grade', x=['Dropout', 'Enrolled', 'Graduated'], y=avg_grades['Curricular_units_2nd_sem_grade'],
                text=[f'{v:.2f}' for v in avg_grades['Curricular_units_2nd_sem_grade']], textposition='auto', marker_color='white'
            ))
            fig.update_layout(get_plotly_dark_theme(), barmode='group', xaxis_title="Status", yaxis_title="Average Grade", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        container = st.container(border=True)
        with container:
            col_course, col_pie = st.columns([4, 1])
            with col_course:
                st.subheader("Dropout Rate by Course")
                course_total = filtered_data.groupby('Course_Label')['Status'].count()
                course_dropout = filtered_data[filtered_data['Status'] == 0].groupby('Course_Label')['Status'].count()
                dropout_rate_course = ((course_dropout / course_total) * 100).fillna(0).round(2).sort_values()

                if not dropout_rate_course.empty:
                    fig = px.bar(
                        x=dropout_rate_course.values, y=dropout_rate_course.index,
                        labels={'y': 'Course', 'x': 'Dropout Rate (%)'},
                        text=[f"{v}%" for v in dropout_rate_course.values],
                        orientation='h', height=600
                    )
                    fig.update_traces(textposition='outside', marker_color='#393939')
                    fig.update_layout(get_plotly_dark_theme(), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No dropout data to display for the selected course/filters.")

            with col_pie:
                create_pie_chart(filtered_data, 'Educational_special_needs', 'Special Needs')
                create_pie_chart(filtered_data, 'Debtor', 'Debtors')
                create_pie_chart(filtered_data, 'Tuition_fees_up_to_date', 'Tuition Fees Paid')

    else:
        st.warning("No data matches the selected filters. Please broaden your selection.")

# --- Prediction Page ---
elif add_selectbox == "Prediction":
    st.title("Student Dropout Prediction")
    st.markdown("Enter the student's details below to predict their dropout likelihood.")

    # Load the pre-trained model and create a reverse mapping for courses
    model = load('model/model.joblib')
    course_list = sorted(data.Course_Label.unique())
    reverse_mapping = {v: k for k, v in data[['Course', 'Course_Label']].drop_duplicates().set_index('Course_Label')['Course'].to_dict().items()}

    with st.form("prediction_form"):
        st.subheader("Student Information")
        # --- Input Fields ---
        col1, col2 = st.columns(2)
        with col1:
            course_label_selected = st.selectbox('Course', course_list, key='course')
            age_selected = st.slider("Age at Enrollment", min_value=17, max_value=70, value=20, step=1)
            admgrade_selected = st.slider("Admission Grade (0-200)", min_value=0.0, max_value=200.0, value=120.0, step=0.1)
            gender_selected = st.radio('Gender', ['Female', 'Male'], horizontal=True)

        with col2:
            grade1_selected = st.slider("1st Semester Grade (0-20)", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
            grade2_selected = st.slider("2nd Semester Grade (0-20)", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
            scholarship_selected = st.radio('Scholarship Holder?', ['No', 'Yes'], horizontal=True)
            debtor_selected = st.radio('Is the student a debtor?', ['No', 'Yes'], horizontal=True)
            
        col3, col4 = st.columns(2)
        with col3:
             special_selected = st.radio('Special Education Needs?', ['No', 'Yes'], horizontal=True)
        with col4:
            tuition_selected = st.radio('Tuition up to date?', ['Yes', 'No'], horizontal=True)

        # Submit Button for the form
        submitted = st.form_submit_button("Run Prediction")

    if submitted:
        # --- Process Inputs and Predict ---
        course_selected = reverse_mapping[course_label_selected]
        time_selected = 0 if course_selected in [9991, 8014] else 1 # Determine attendance time from course
        
        user_data = {
            'Course': course_selected,
            'Daytime_evening_attendance': time_selected,
            'Admission_grade': admgrade_selected,
            'Educational_special_needs': 1 if special_selected == 'Yes' else 0,
            'Debtor': 1 if debtor_selected == 'Yes' else 0,
            'Tuition_fees_up_to_date': 1 if tuition_selected == 'Yes' else 0,
            'Gender': 1 if gender_selected == 'Male' else 0,
            'Scholarship_holder': 1 if scholarship_selected == 'Yes' else 0,
            'Age_at_enrollment': age_selected,
            'Curricular_units_1st_sem_grade': grade1_selected,
            'Curricular_units_2nd_sem_grade': grade2_selected
        }

        X_new = pd.DataFrame([user_data])
        prediction = model.predict(X_new)[0]
        prediction_proba = model.predict_proba(X_new)[0]

        st.subheader("Prediction Result")
        if prediction == 0:
            probability_score = prediction_proba[0] * 100
            st.error(f"**Prediction: Student is likely to DROPOUT** (Confidence: {probability_score:.2f}%)")
            st.warning("Action Recommended: Consider advising the student to connect with academic support services or a student counselor.")
        else: # prediction is 1 (Enrolled or Graduated)
            probability_score = prediction_proba[1] * 100
            st.success(f"**Prediction: Student is NOT likely to dropout** (Confidence: {probability_score:.2f}%)")
            st.info("The student appears to be on a positive academic track.")

