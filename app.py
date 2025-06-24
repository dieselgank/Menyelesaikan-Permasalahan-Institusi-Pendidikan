import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from joblib import load
import plotly.graph_objects as go

# Set Streamlit page configuration to wide layout
st.set_page_config(layout="wide")

# --- Data Loading and Preprocessing ---
# Load the dataset
data = pd.read_csv("dataset/df_new.csv", delimiter=",")

# Create subsets based on 'Status' for potential future use (though not directly used in current filtering logic)
data_0 = data.loc[data['Status'] == 0] # Dropout
data_1 = data.loc[data['Status'] == 1] # Enrolled
data_2 = data.loc[data['Status'] == 2] # Graduated

# Define a mapping for Course IDs to human-readable labels
category_mapping = {
    33: 'Biofuel Production Technologies',
    171: 'Animation and Multimedia Design',
    8014: 'Social Service (evening attendance)',
    9003: 'Agronomy',
    9070: 'Communication Design',
    9085: 'Veterinary Nursing',
    9119: 'Informatics Engineering',
    9130: 'Equinculture',
    9147: 'Management',
    9238: 'Social Service',
    9254: 'Tourism',
    9500: 'Nursing',
    9556: 'Oral Hygiene',
    9670: 'Advertising and Marketing Management',
    9773: 'Journalism and Communication',
    9853: 'Basic Education',
    9991: 'Management (evening attendance)'
}
# Apply the mapping to create a new 'Course_Label' column
data['Course_Label'] = data['Course'].replace(category_mapping)

# --- Sidebar Navigation ---
add_selectbox = st.sidebar.selectbox(
    "Choose a page",
    ("Dashboard", "Prediction")
)

# --- Custom HTML Styling Functions ---
def add_rating(content):
    """
    Generates an HTML string for displaying key metrics with a stylish card-like appearance.
    """
    return f"""
        <div style='
            height: auto;
            border: 1px solid #ddd; /* Lighter, more subtle border */
            border-radius: 8px; /* Slightly more rounded corners */
            font-size: 25px;
            padding: 20px; /* Adjusted padding */
            background-color: #f9f9f9; /* Off-white background */
            text-align: center;
            display: flex;
            flex-direction: column; /* Stack content vertically */
            justify-content: center;
            align-items: center;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1); /* Subtle shadow */
            '>
            {content}
        </div>
        """

def add_card(content):
    """
    Generates an HTML string for displaying general information cards.
    """
    return f"""
        <div style='
            height: auto;
            font-size: auto;
            border: 1px solid #ddd; /* Lighter, more subtle border */
            border-radius: 8px; /* Slightly more rounded corners */
            padding: 10px;
            margin-bottom: 10px;
            background-color: #f9f9f9; /* Off-white background */
            text-align: center;
            display: flex;
            flex-direction: column; /* Stack content vertically */
            justify-content: center;
            align-items: center;
            line-height: 1.4; /* Adjusted line height for better readability */
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1); /* Subtle shadow */
            '>
            {content}
        </div>
        """

def create_pie_chart(column, title):
    """
    Creates and displays a pie chart for a given column.
    Handles cases where data might be insufficient for plotting.
    """
    try:
        # Filter out NaN values before counting
        filtered_kelas = kelas[column].dropna()
        value_counts = filtered_kelas.value_counts()

        if value_counts.empty:
            st.write(f"No data available for '{title}' after applying filters.")
            return

        names = [str(bool(val)) for val in value_counts.index] # Convert 0/1 to False/True strings
        colors = ['#393939', 'white'] # Consistent color scheme

        fig = px.pie(
            values=value_counts,
            names=names,
            title=title,
            color_discrete_sequence=colors
        )
        fig.update_layout(
            height=250, # Slightly increased height for better visibility
            margin=dict(l=0, r=10, t=70, b=10),
            title=dict(
                x=0.5, # Center the title
                font=dict(size=15),
            ),
            plot_bgcolor='rgba(0,0,0,0)', # Transparent background for the plot area
            paper_bgcolor='rgba(0,0,0,0)', # Transparent background for the paper area
            font=dict(color='white') # Ensure text color is white for readability
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.write(f"An error occurred while creating the pie chart for '{title}': {e}")


# --- Dashboard Page Logic ---
if add_selectbox == "Dashboard":
    st.title('Jaya Jaya Institute Student Performance Dashboard')

    # --- Filter Section ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Status filter
        status_options = ['None', 'Dropout', 'Not Dropout']
        selected_status = st.selectbox('Select Status', status_options, key='initial_status')

        # Nested filter for 'Not Dropout' status
        if selected_status == 'Not Dropout':
            # st.session_state['split_columns'] = True # This state is not directly used for rendering columns here
            not_dropout_options = ['None', 'Enrolled', 'Graduated']
            selected_not_dropout_type = st.selectbox('Select Type of Not Dropout', not_dropout_options, key='not_dropout_type')
            if selected_not_dropout_type == 'Enrolled':
                actual_status_filter = 1
            elif selected_not_dropout_type == 'Graduated':
                actual_status_filter = 2
            else: # 'None' for not_dropout_options means combine Enrolled and Graduated
                actual_status_filter = 'Not Dropout' # Special string to handle combined filter
        elif selected_status == 'Dropout':
            actual_status_filter = 0
        else: # 'None' for main status filter
            actual_status_filter = 'None'

    with col2:
        # Course filter
        course_list = ['None'] + sorted(list(data.Course_Label.unique()))
        selected_course = st.selectbox('Select Course', course_list)

    with col3:
        # Attendance time filter
        time_options = ['None', 'Daytime', 'Evening']
        selected_time_label = st.selectbox('Select Attendance Time', time_options)
        
        # Convert label to numerical value for filtering
        if selected_time_label == 'Daytime':
            selected_time_value = 1
        elif selected_time_label == 'Evening':
            selected_time_value = 0
        else:
            selected_time_value = 'None'

    with col4:
        # Gender filter
        gender_options = ['None', 'Male', 'Female']
        selected_gender_label = st.selectbox('Select Gender', gender_options)

        # Convert label to numerical value for filtering
        if selected_gender_label == 'Male':
            selected_gender_value = 1
        elif selected_gender_label == 'Female':
            selected_gender_value = 0
        else:
            selected_gender_value = 'None'

    # --- Apply Filters to DataFrame ---
    kelas = data.copy() # Start with a copy of the original data

    # Apply status filter
    if actual_status_filter == 'None':
        pass # No status filter applied
    elif actual_status_filter == 'Not Dropout':
        kelas = kelas.loc[kelas['Status'].isin([1, 2])] # Combine Enrolled (1) and Graduated (2)
    else:
        kelas = kelas.loc[kelas['Status'] == actual_status_filter]

    # Apply course filter
    if selected_course != "None":
        kelas = kelas.loc[kelas['Course_Label'] == selected_course]

    # Apply attendance time filter
    if selected_time_value != "None":
        kelas = kelas.loc[kelas['Daytime_evening_attendance'] == selected_time_value]

    # Apply gender filter
    if selected_gender_value != "None":
        kelas = kelas.loc[kelas['Gender'] == selected_gender_value]

    # --- Key Performance Indicators (KPIs) ---
    containerA = st.container(border=True)
    containerB = st.container(border=True)

    colDr, colSt = st.columns([1, 2])

    with containerA:
        with colDr:
            total_students = kelas['Status'].count() # Count of rows in the filtered data

            if total_students > 0:
                dropout_count = kelas.loc[kelas['Status'] == 0].shape[0]
                enrolled_count = kelas.loc[kelas['Status'] == 1].shape[0]
                graduated_count = kelas.loc[kelas['Status'] == 2].shape[0]

                dropout_rate = f"{round((dropout_count / total_students) * 100, 2)}%"
                enrolled_rate = f"{round((enrolled_count / total_students) * 100, 2)}%"
                graduation_rate = f"{round((graduated_count / total_students) * 100, 2)}%"

                st.markdown(add_rating(f"<b>Dropout Rate</b><br>{dropout_rate}"), unsafe_allow_html=True)
                # You can add enrolled and graduation rates here if desired in a similar card
                # col1, col2 = st.columns(2)
                # with col1:
                #     st.markdown(add_rating(f"<b>Enrolled Rate</b><br>{enrolled_rate}"), unsafe_allow_html=True)
                # with col2:
                #     st.markdown(add_rating(f"<b>Graduation Rate</b><br>{graduation_rate}"), unsafe_allow_html=True)
            else:
                st.markdown(add_rating("<b>Dropout Rate</b><br>N/A"), unsafe_allow_html=True)

    with containerB:
        with colSt:
            total_students_in_filtered_data = kelas['Status'].count()
            st.markdown(add_card(f"<b>Total Students</b><br>{total_students_in_filtered_data}"), unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                data_do = kelas.loc[kelas['Status'] == 0].shape[0]
                st.markdown(add_card(f"<b>Dropped out</b><br>{data_do}"), unsafe_allow_html=True)
            with col2:
                data_enrolled = kelas.loc[kelas['Status'] == 1].shape[0]
                st.markdown(add_card(f"<b>Enrolled</b><br>{data_enrolled}"), unsafe_allow_html=True)
            with col3:
                data_graduated = kelas.loc[kelas['Status'] == 2].shape[0]
                st.markdown(add_card(f"<b>Graduated</b><br>{data_graduated}"), unsafe_allow_html=True)

    # --- Charts Section ---
    col1, col2 = st.columns(2)

    # Determine the grouping column based on selected status filter
    if selected_status == "None":
        grouper = "Status"
    elif selected_status == 'Not Dropout':
        grouper = "Status_New" # Assuming Status_New is 1 for enrolled/graduated, 0 for dropout
        # Note: 'Status_New' column definition is not in the provided snippet.
        # If 'Status_New' doesn't exist or is not correctly defined for this use case,
        # it might cause issues. Assuming it aggregates 1 and 2 into a single category.
    else:
        # If a specific status (0, 1, or 2) is selected, no further grouping by status is needed,
        # but the `grouper` variable is still used for x-axis labeling in charts.
        grouper = "Status" # Or a direct string like 'Selected Status'

    with col1:
        st.subheader('Scholarship Holder by Status')
        try:
            # Filter for actual scholarship holders within the filtered 'kelas' DataFrame
            scholarship_data = kelas[kelas['Scholarship_holder'] == 1]
            if not scholarship_data.empty:
                # Group by 'Status' and count scholarship holders
                # Using 'Status' directly as it's the most granular and correct for this chart
                scholarship_counts = scholarship_data.groupby('Status').size()

                if not scholarship_counts.empty:
                    # Determine the status with the maximum scholarship holders for coloring
                    max_scholarship_status = scholarship_counts.idxmax()

                    # Define colors for bars: 'white' for the max, '#393939' for others
                    # Map numerical status to labels for x-axis ticks
                    status_labels = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduated'}
                    x_labels = [status_labels.get(s, str(s)) for s in scholarship_counts.index]
                    bar_colors = ['white' if s == max_scholarship_status else '#393939' for s in scholarship_counts.index]

                    bars = go.Bar(
                        x=x_labels, # Use labels for x-axis
                        y=scholarship_counts.values,
                        marker=dict(color=bar_colors, line=dict(color='white', width=1)),
                        text=scholarship_counts.values,
                        textposition='auto'
                    )

                    layout = {
                        'xaxis': {'title': 'Student Status', 'tickfont': {'color': 'white'}, 'color': 'white', 'showline': True, 'linecolor': 'white', 'linewidth': 1},
                        'yaxis': {'title': 'Number of Scholarship Holders', 'tickfont': {'color': 'white'}, 'color': 'white', 'showline': True, 'linecolor': 'white', 'linewidth': 1},
                        'plot_bgcolor': 'rgba(0,0,0,0)', # Transparent background
                        'paper_bgcolor': 'rgba(0,0,0,0)', # Transparent background
                        'font': {'color': 'white'},
                        'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40},
                        'xaxis_showline': True,
                        'yaxis_showline': True
                    }
                    fig = go.Figure(data=[bars], layout=layout)
                    st.plotly_chart(fig)
                else:
                    st.write("No scholarship holders found for the selected filters to display this chart.")
            else:
                st.write("No scholarship holders found for the selected filters to display this chart.")

        except Exception as e:
            st.write(f"An error occurred while creating the 'Scholarship Holder by Status' chart: {e}")

    with col2:
        st.subheader('Average Grade per Semester')
        try:
            if not kelas.empty:
                # Group by 'Status' for average grades
                avg_1st_sem = kelas.groupby('Status')['Curricular_units_1st_sem_grade'].mean()
                avg_2nd_sem = kelas.groupby('Status')['Curricular_units_2nd_sem_grade'].mean()

                if not avg_1st_sem.empty and not avg_2nd_sem.empty:
                    # Calculate difference and percentage difference
                    diff = avg_2nd_sem - avg_1st_sem
                    # Avoid division by zero if sum is zero
                    percentage_diff = (diff / (avg_1st_sem.replace(0, np.nan) + avg_2nd_sem.replace(0, np.nan))) * 100
                    percentage_diff = percentage_diff.fillna(0) # Handle NaN results from division by zero

                    # Map numerical status to labels for x-axis ticks
                    status_labels = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduated'}
                    x_labels_grades = [status_labels.get(s, str(s)) for s in avg_1st_sem.index]

                    bars_1st_sem = go.Bar(
                        x=x_labels_grades,
                        y=avg_1st_sem,
                        name='1st Semester',
                        marker=dict(color='#393939', line=dict(color='white', width=1)),
                        text=[f"{value:.2f}" for value in avg_1st_sem],
                        textposition='auto'
                    )

                    bars_2nd_sem = go.Bar(
                        x=x_labels_grades,
                        y=avg_2nd_sem,
                        name='2nd Semester',
                        marker=dict(color='white', line=dict(color='white', width=1)),
                        text=[f"{value:.2f}" for value in avg_2nd_sem],
                        textposition='auto'
                    )

                    layout = {
                        'xaxis': {
                            'title': 'Student Status',
                            'tickfont': {'color': 'white'},
                            'color': 'white',
                            'showline': True,
                            'linecolor': 'white',
                            'linewidth': 1
                        },
                        'yaxis': {
                            'title': 'Average Grade',
                            'tickfont': {'color': 'white'},
                            'color': 'white',
                            'showline': True,
                            'linecolor': 'white',
                            'linewidth': 1
                        },
                        'plot_bgcolor': 'rgba(0,0,0,0)', # Transparent background
                        'paper_bgcolor': 'rgba(0,0,0,0)', # Transparent background
                        'font': {'color': 'white'},
                        'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40},
                        'xaxis_showline': True,
                        'yaxis_showline': True,
                        'barmode': 'group'
                    }

                    fig_semesters = go.Figure(data=[bars_1st_sem, bars_2nd_sem], layout=layout)

                    for i, status in enumerate(avg_1st_sem.index):
                        # Add annotation only if there's valid percentage data
                        if not pd.isna(percentage_diff[status]):
                            fig_semesters.add_annotation(
                                x=x_labels_grades[i],
                                y=max(avg_1st_sem[status], avg_2nd_sem[status]) + 2,
                                text=f"Diff: {diff[status]:.2f} ({percentage_diff[status]:.2f}%)",
                                showarrow=False,
                                font=dict(color="white", size=10), # Smaller font for annotations
                                align='center'
                            )

                    st.plotly_chart(fig_semesters)
                else:
                    st.write("Insufficient grade data for the selected filters to display this chart.")
            else:
                st.write("No data available for the selected filters to display this chart.")

        except Exception as e:
            st.write(f"An error occurred while creating the 'Average Grade per Semester' chart: {e}")

    # --- Dropout Rate by Course and other distributions ---
    container_bottom = st.container(border=True) # Renamed to avoid conflict with `container` variable
    colA, colB = st.columns([4, 1])

    with container_bottom:
        with colA:
            st.subheader("Dropout Rate by Course")

            # Create a copy to avoid modifying the original 'kelas' for this specific chart
            course_kls = kelas.copy()

            # Ensure 'Course' column is mapped to labels if not already done for this specific context
            # It's already done at the beginning, but explicitly mapping here for clarity if needed.
            # No, 'Course_Label' is used, so no re-mapping needed on 'Course' itself.

            # Filter for dropout students (Status == 0) and non-dropout students (Status > 0)
            data_do_chart = course_kls[course_kls['Status'] == 0]
            data_notdo_chart = course_kls[course_kls['Status'].isin([1, 2])]

            # Group by Course_Label and sum counts
            course_do_counts = data_do_chart.groupby('Course_Label').size()
            course_notdo_counts = data_notdo_chart.groupby('Course_Label').size()

            # Align indices of the two series to ensure correct calculations
            # Fill NaN with 0 for courses that only have dropouts or non-dropouts
            all_courses = pd.Index(list(course_do_counts.index) + list(course_notdo_counts.index)).unique()
            course_do_counts = course_do_counts.reindex(all_courses, fill_value=0)
            course_notdo_counts = course_notdo_counts.reindex(all_courses, fill_value=0)

            try:
                # Calculate total students for each course and dropout rate
                total_per_course = course_do_counts + course_notdo_counts
                # Avoid division by zero by replacing 0 with NaN, then filling with 0 after division
                dropout_rate_by_course = (course_do_counts / total_per_course.replace(0, np.nan)) * 100
                dropout_rate_by_course = dropout_rate_by_course.fillna(0) # If total_per_course was 0, dropout_rate is 0

                # Sort for better visualization
                a_sorted = dropout_rate_by_course.sort_values(ascending=True)

                if not a_sorted.empty:
                    # Determine the course with the highest dropout rate for coloring
                    max_dropout_course = a_sorted.idxmax()

                    fig = px.bar(
                        x=a_sorted.values,
                        y=a_sorted.index,
                        labels={'y': 'Course', 'x': 'Dropout Rate (%)'},
                        text=[f"{value:.2f}%" for value in a_sorted.values], # Format text for percentages
                        color=a_sorted.index.map(lambda x: 'white' if x == max_dropout_course else '#393939'), # Dynamic coloring
                        color_discrete_map={'white': 'white', '#393939': '#393939'}, # Explicitly map colors
                        height=600,
                        orientation='h' # Ensure horizontal bars
                    )

                    fig.update_traces(
                        textposition='outside',
                        marker_line_width=1, # Add border to bars
                        marker_line_color='white'
                    )

                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', # Transparent background
                        paper_bgcolor='rgba(0,0,0,0)', # Transparent background
                        font=dict(color='white'),
                        xaxis=dict(
                            tickfont=dict(color='white'),
                            linecolor='white',
                            showgrid=False # Remove grid lines
                        ),
                        yaxis=dict(
                            tickfont=dict(color='white'),
                            linecolor='white',
                            showgrid=False # Remove grid lines
                        ),
                        showlegend=False,
                        title_font_size=20 # Adjust title font size
                    )

                    st.plotly_chart(fig)
                else:
                    st.write("No course-specific dropout data available for the selected filters.")

            except Exception as e:
                st.write(f"An error occurred while creating the 'Dropout Rate by Course' chart: {e}")

        with colB:
            # Pie charts for binary features
            create_pie_chart('Educational_special_needs', 'Educational Special Needs<br>Distribution')
            create_pie_chart('Debtor', 'Debtor Distribution')
            create_pie_chart('Tuition_fees_up_to_date', 'Tuition Fees Up to Date<br>Distribution')

    # --- Age Distribution ---
    container_age = st.container(border=True)
    colA_age, colB_age = st.columns([4, 1])
    colors_age = ['#393939'] # Using a single color for histogram bars

    with container_age:
        with colA_age:
            try:
                if not kelas.empty:
                    fig = px.histogram(
                        kelas,
                        x='Age_at_enrollment',
                        title='Age at Enrollment Distribution',
                        color_discrete_sequence=colors_age # Use the defined color
                    )

                    fig.update_layout(
                        title=dict(
                            text='Age at Enrollment Distribution',
                            font=dict(size=24, color='white') # Set title color
                        ),
                        plot_bgcolor='rgba(0,0,0,0)', # Transparent background
                        paper_bgcolor='rgba(0,0,0,0)', # Transparent background
                        font=dict(color='white'),
                        xaxis=dict(
                            title='Age at Enrollment',
                            tickfont=dict(color='white'),
                            linecolor='white',
                            showgrid=False
                        ),
                        yaxis=dict(
                            title='Number of Students',
                            tickfont=dict(color='white'),
                            linecolor='white',
                            showgrid=False
                        ),
                        bargap=0.1 # Gap between bars for better visual separation
                    )
                    st.plotly_chart(fig)
                else:
                    st.write("No age data available for the selected filters to display this chart.")
            except Exception as e:
                st.write(f"An error occurred while creating the 'Age at Enrollment Distribution' chart: {e}")

        with colB_age:
            if not kelas.empty:
                max_age = int(kelas['Age_at_enrollment'].max())
                mean_age = round(kelas['Age_at_enrollment'].mean(), 1)
                min_age = int(kelas['Age_at_enrollment'].min())
                st.markdown(add_card(f"<b>Minimum Age</b><br>{min_age}"), unsafe_allow_html=True)
                st.markdown(add_card(f"<b>Average Age</b><br>{mean_age}"), unsafe_allow_html=True)
                st.markdown(add_card(f"<b>Maximum Age</b><br>{max_age}"), unsafe_allow_html=True)
            else:
                st.markdown(add_card("<b>Minimum Age</b><br>N/A"), unsafe_allow_html=True)
                st.markdown(add_card("<b>Average Age</b><br>N/A"), unsafe_allow_html=True)
                st.markdown(add_card("<b>Maximum Age</b><br>N/A"), unsafe_allow_html=True)


# --- Prediction Page Logic ---
if add_selectbox == "Prediction":
    st.subheader("Predict Student Dropout Status")

    # Reverse mapping for course labels to IDs for prediction model input
    reverse_mapping = {v: k for k, v in category_mapping.items()}

    # Course selection dropdown
    course_list_pred = sorted(list(data.Course_Label.unique()))
    course_selected_label = st.selectbox('Course', ['None', *course_list_pred])

    # Store selected course in session state
    if course_selected_label == 'None':
        st.error("Please select a valid course to proceed with prediction.")
        course_selected_id = None # Set to None if no course is selected
    else:
        st.session_state.course_selected_label = course_selected_label
        course_selected_id = reverse_mapping[course_selected_label]

    # Automatic determination of 'Daytime_evening_attendance' based on course
    # This logic assumes specific courses are always evening attendance
    if course_selected_id in [9991, 8014]:
        time_selected = 0 # Evening attendance
    else:
        time_selected = 1 # Daytime attendance (default)
    st.info(f"Attendance time for selected course is automatically set to: {'Evening' if time_selected == 0 else 'Daytime'}")

    # Admission grade input
    admgrade_selected = st.number_input(
        "Admission Grade (0.0 to 200.0)",
        value=0.0,
        step=0.1,
        min_value=0.0,
        max_value=200.0,
        format="%.1f" # Format to one decimal place
    )
    # admgrade_selected is already rounded by format in st.number_input

    # Gender and Age inputs in two columns
    colGender, colAge = st.columns(2)
    with colGender:
        gender_list = ['Male', 'Female']
        gender_selected_label = st.selectbox('Gender', gender_list)
        gender_selected_value = 1 if gender_selected_label == "Male" else 0

    with colAge:
        age_selected = st.number_input(
            "Age at Enrollment (17 to 70)",
            step=1,
            min_value=17,
            max_value=70,
            value=17 # Default value
        )

    # Binary feature inputs (Special needs, Debtor) in two columns
    bool1, bool2 = st.columns(2)
    with bool1:
        special_list = ['No', 'Yes'] # 'No' as default/first option
        special_selected_label = st.radio('Has Special Education Needs?', special_list)
        special_selected_value = 1 if special_selected_label == "Yes" else 0

    with bool2:
        debtor_list = ['No', 'Yes'] # 'No' as default/first option
        debtor_selected_label = st.radio('Is a Debtor?', debtor_list)
        debtor_selected_value = 1 if debtor_selected_label == "Yes" else 0

    # Binary feature inputs (Tuition up to date, Scholarship holder) in two columns
    bool3, bool4 = st.columns(2)
    with bool3:
        tuition_list = ['Yes', 'No'] # 'Yes' as default/first option
        tuition_selected_label = st.radio('Are Tuition Fees Up to Date?', tuition_list)
        tuition_selected_value = 1 if tuition_selected_label == "Yes" else 0

    with bool4:
        scholarship_list = ['No', 'Yes'] # 'No' as default/first option
        scholarship_selected_label = st.radio('Is a Scholarship Holder?', scholarship_list)
        scholarship_selected_value = 1 if scholarship_selected_label == "Yes" else 0

    # Semester grades inputs in two columns
    grade1, grade2 = st.columns(2)
    with grade1:
        grade1_selected = st.number_input(
            "First Semester Grade (0.0 to 20.0)",
            value=0.0,
            step=0.1,
            min_value=0.0,
            max_value=20.0,
            format="%.2f" # Format to two decimal places
        )
        # grade1_selected is already rounded by format in st.number_input

    with grade2:
        grade2_selected = st.number_input(
            "Second Semester Grade (0.0 to 20.0)",
            value=0.0,
            step=0.1,
            min_value=0.0,
            max_value=20.0,
            format="%.2f" # Format to two decimal places
        )
        # grade2_selected is already rounded by format in st.number_input

    # Custom CSS for the Predict button
    st.markdown(
        '<style>div.stButton > button {margin: 0 auto; display: block; background: #393939; color: white; border: none; border-radius: 5px; padding: 10px 20px; font-size: 18px; cursor: pointer; transition: background-color 0.3s ease;}</style>',
        unsafe_allow_html=True
    )
    # Add hover effect for the button
    st.markdown(
        '<style>div.stButton > button:hover {background-color: #555555;}</style>',
        unsafe_allow_html=True
    )

    # Predict button
    button_predict = st.button("Predict Student Status", key='custom_button')

    if button_predict:
        if course_selected_id is None:
            st.error("Please select a valid course before predicting.")
        else:
            try:
                # Load the pre-trained model
                model = load('model/model.joblib')

                # Prepare user input data for prediction
                user_data = {
                    'Course': [course_selected_id],
                    'Daytime_evening_attendance': [time_selected],
                    'Admission_grade': [admgrade_selected],
                    'Educational_special_needs': [special_selected_value],
                    'Debtor': [debtor_selected_value],
                    'Tuition_fees_up_to_date': [tuition_selected_value],
                    'Gender': [gender_selected_value],
                    'Scholarship_holder': [scholarship_selected_value],
                    'Age_at_enrollment': [age_selected],
                    'Curricular_units_1st_sem_grade': [grade1_selected],
                    'Curricular_units_2nd_sem_grade': [grade2_selected]
                }

                # Create a DataFrame from user input
                X_new = pd.DataFrame(user_data)

                # Make prediction
                predictions = model.predict(X_new)

                st.subheader("Prediction Result")
                if predictions == 0:
                    st.error("Based on the provided information, the student is predicted to be a **Dropout**.")
                elif predictions == 1:
                    st.success("Based on the provided information, the student is **NOT** predicted to be a dropout (likely to be Enrolled or Graduated).")

            except FileNotFoundError:
                st.error("Error: The prediction model file 'model/model.joblib' was not found. Please ensure it's in the correct directory.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
