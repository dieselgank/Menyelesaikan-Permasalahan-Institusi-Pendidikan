import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from joblib import load
import plotly.graph_objects as go

st. set_page_config(layout="wide") #wide or centered

data = pd.read_csv("dataset/df_new.csv", delimiter=",")
data_0 = data.loc[data['Status']==0]
data_1 = data.loc[data['Status']==1]
data_2 = data.loc[data['Status']==2]

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
data['Course_Label'] = data['Course'].replace(category_mapping)

add_selectbox = st.sidebar.selectbox(
    "Choose a page",
    ("Dashboard", "Prediction")
)

def add_rating(content):
    # return f"<div style='border: 2px solid #ccc; border-radius: 5px; padding: 10px;'>{content}</div>"
    return f"""
        <div style='
            height: auto;
            border: 2px solid #ccc;
            border-radius: 5px;
            font-size: 25px;
            padding-bottom: 38px;
            padding-top: 38px;
            background-color: #fffff;
            text-align: center; /* Center text horizontally */
            display: flex;
            justify-content: center;
            align-items: center;
            '>{content}</div>
        """

def add_card(content):
    # return f"<div style='border: 2px solid #ccc; border-radius: 5px; padding: 10px;'>{content}</div>"
    return f"""
        <div style='
            height: auto;
            font-size: auto;
            border: 2px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
            background-color: #fffff;
            text-align: center; /* Center text horizontally */
            display: flex; /* Center text vertically */
            justify-content: center; /* Center text vertically */
            align-items: center;
            line-height: 70px;
            '>{content}</div>
        """

def create_pie_chart(column, title):
    try:
        value_counts = kelas[column].value_counts()
        if len(value_counts) > 1:
            names = [False, True]
        else:
            if value_counts.index==1:
                names = [True]
            elif value_counts.index==0:
                names= [False]
        colors = ['white', '#393939']
        fig = px.pie(
            values=value_counts,
            names=names,
            title=title,
            color_discrete_sequence=colors)
        fig.update_layout(
            height=200,
            margin=dict(l=0, r=10, t=70, b=10),
            title=dict(
                x=0,
                font=dict(size=15),
            ),
        )
        st.plotly_chart(fig)
    except UnboundLocalError as e:
        st.write("No data available to display.")

if add_selectbox == "Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # status_list = list(data.Status.unique())[::-1]
        # status_list.sort()
        # status_list.insert(0,"None")
        status_list = ['None', 'Dropout', 'Not Dropout']
        selected_status = st.selectbox('Select status', (status_list), key='initial_status')
        if selected_status == 'Dropout':
            selected_status = 0
        elif selected_status == 'Not Dropout':
            st.session_state['split_columns'] = True
            status_list = ['None', 'Enrolled', 'Graduated']
            selected_status = st.selectbox('Select type of Not Dropout', (status_list), key='not_dropout_type')
            if selected_status == 'None':
                selected_status = 'Not Dropout'
            if selected_status == 'Enrolled':
                selected_status = 1
            if selected_status == 'Graduated':
                selected_status = 2
            kelas_selected_status = data[data.Status == selected_status]

    with col2:
        course_list = list(data.Course_Label.unique())[::-1]
        course_list.sort()
        course_list.insert(0,"None")
        selected_course = st.selectbox('Select course', (course_list))
        kelas_selected_course = data[data.Course_Label == selected_course]

    with col3:
        time_list = ['None', 'Daytime', 'Evening']
        # time_list = ['None']
        # if (data['Daytime_evening_attendance'] == 0).any():
        #     time_list.append("Evening")
        # if (data['Daytime_evening_attendance'] == 1).any():
        #     time_list.append("Daytime")

        selected_time = st.selectbox('Select attendance time', (time_list))
        kelas_selected_time = data[data.Daytime_evening_attendance == selected_time]
        if selected_time == 'Daytime':
            selected_time = 1
        elif selected_time == 'Evening':
            selected_time = 0

    with col4:
        gender_list = ['None', 'Male', 'Female']
        selected_gender = st.selectbox('Select gender', (gender_list))
        kelas_selected_gender = data[data.Course_Label == selected_gender]
        if selected_gender=='Male':
            selected_gender=1
        elif selected_gender=='Female':
            selected_gender=0
    
    if selected_status=='None':
        kelas = data
    elif selected_status=='Not Dropout':
        kelas = data.loc[data['Status_New']==1]
    else:
        kelas = data.loc[data['Status']==selected_status]

    if selected_course=="None":
        kelas = kelas
    else:
        kelas = kelas.loc[kelas['Course_Label']==selected_course]

    if selected_time=="None":
        kelas = kelas
    else:
        kelas = kelas.loc[kelas['Daytime_evening_attendance']==selected_time]

    if selected_gender=="None":
        kelas = kelas
    else:
        kelas = kelas.loc[kelas['Gender']==selected_gender]

    st.title('Jaya Jaya Institute Student Performance Dashboard')

    
    #===============================================
    containerA = st.container(border=True)
    containerB = st.container(border=True)

    colDr, colSt = st.columns([1,2])

    
    with containerA:
        with colDr:
            dropout_rate = str(round((kelas['Status_0'].sum()/(kelas['Status_0'].sum()+kelas['Status_1'].sum()+kelas['Status_2'].sum()))*100,3))+"%"
            enrolled_rate = str(round((kelas['Status_1'].sum()/(kelas['Status_0'].sum()+kelas['Status_1'].sum()+kelas['Status_2'].sum()))*100,3))+"%"
            graduation_rate = str(round((kelas['Status_2'].sum()/(kelas['Status_0'].sum()+kelas['Status_1'].sum()+kelas['Status_2'].sum()))*100,3))+"%"
            # st.subheader(f"Dropout Rate: {dropout_rate}")
            # st.markdown(f"{dropout_rate}")
            st.markdown(add_rating(f"<b>Dropout Rate</b><br>{dropout_rate}"), unsafe_allow_html=True)
            # col1, col2 = st.columns(2)
            # with col1:
            #     st.markdown(add_rating(f"<b>Enrolled Rate</b><br>{enrolled_rate}"), unsafe_allow_html=True)
            # with col2:
            #     st.markdown(add_rating(f"<b>Graduation Rate</b><br>{graduation_rate}"), unsafe_allow_html=True)
            # st.subheader(dropout_rate)
            
    with containerB:   
        with colSt:
            # st.subheader('Total Students')

            data_total = kelas['Status_0'].sum()+kelas['Status_1'].sum()+kelas['Status_2'].sum()
            st.markdown(add_card(f"<b>Total students</b><br>{data_total}"), unsafe_allow_html=True)
 
            col1, col2, col3 = st.columns(3)

            with col1:
                    data_do = kelas['Status_0'].sum()
                    st.markdown(add_card(f"<b>Dropped out student</b><br>{data_do}"), unsafe_allow_html=True)

            with col2:
                    data_enrolled = kelas['Status_1'].sum()
                    st.markdown(add_card(f"<b>Enrolled student</b><br>{data_enrolled}"), unsafe_allow_html=True)

            with col3:
                    data_graduated = kelas['Status_2'].sum()
                    st.markdown(add_card(f"<b>Graduated student</b><br>{data_graduated}"), unsafe_allow_html=True)


    #------------------------

    col1, col2 = st.columns(2)

    if selected_status=="None":
        grouper = "Status"
    elif selected_status=='Not Dropout':
        grouper = "Status_New"
    else:
        grouper = "Status_"+str(selected_status)

    with col1:
        st.subheader('Scholarship Holder by Status')
        
        a = kelas.groupby(grouper)['Scholarship_holder'].sum()

        try:
            maxA = a.idxmax()

            colors = ['#393939' if b != maxA else 'white' for b in a.index]
            bars = go.Bar(x=a.index, y=a, marker=dict(color=colors, line=dict(color='white', width=1)), text=a, textposition='auto')

            layout = {
                'xaxis': {'title': 'Status', 'tickfont': {'color': 'white'}, 'color': 'white', 'showline': True, 'linecolor': 'white', 'linewidth': 1},
                'yaxis': {'title': 'Number of Student', 'tickfont': {'color': 'white'}, 'color': 'white', 'showline': True, 'linecolor': 'white', 'linewidth': 1},
                'plot_bgcolor': '#111111',
                'paper_bgcolor': '#111111',
                'font': {'color': 'white'},
                'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40},
                'xaxis_showline': True,
                'yaxis_showline': True
            }
            fig = go.Figure(data=[bars], layout=layout)

            if grouper == 'Status':
                fig.update_xaxes(tickvals=a.index, ticktext=['Dropout', 'Enrolled', 'Graduated'], tickangle=0)

            st.plotly_chart(fig)
        
        except ValueError as e:
            st.write("No data available to display.")

    #------------------------
    
    with col2:
        st.subheader('Average Grade per Semester')

        try:
            avg_1st_sem = kelas.groupby(grouper)['Curricular_units_1st_sem_grade'].mean()
            avg_2nd_sem = kelas.groupby(grouper)['Curricular_units_2nd_sem_grade'].mean()

            diff = avg_2nd_sem - avg_1st_sem
            percentage_diff = (diff/(avg_1st_sem+avg_2nd_sem))*100

            bars_1st_sem = go.Bar(
                x=avg_1st_sem.index, 
                y=avg_1st_sem, 
                name='1st Semester',
                marker=dict(color='#393939', line=dict(color='white', width=1)),
                text=[f"{value:.2f}" for value in avg_1st_sem],
                textposition='auto'
            )

            bars_2nd_sem = go.Bar(
                x=avg_2nd_sem.index, 
                y=avg_2nd_sem, 
                name='2nd Semester',
                marker=dict(color='white', line=dict(color='white', width=1)),
                text=[f"{value:.2f}" for value in avg_2nd_sem],
                textposition='auto'
            )

            layout = {
                'xaxis': {
                    'title': 'Status', 
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
                'plot_bgcolor': '#111111',
                'paper_bgcolor': '#111111',
                'font': {'color': 'white'},
                'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40},
                'xaxis_showline': True,
                'yaxis_showline': True,
                'barmode': 'group'  # Group the bars side by side
            }

            fig_semesters = go.Figure(data=[bars_1st_sem, bars_2nd_sem], layout=layout)

            for i, status in enumerate(avg_1st_sem.index):
                fig_semesters.add_annotation(
                    x=status,
                    y=max(avg_1st_sem[status], avg_2nd_sem[status]) + 2,  # Position above the tallest bar
                    text=f"Difference: {diff[status]:.2f} ({percentage_diff[status]:.2f}%)",
                    showarrow=False,
                    font=dict(color="white"),
                    align='center'
                )

            if grouper == 'Status':
                fig_semesters.update_xaxes(tickvals=avg_1st_sem.index, ticktext=['Dropout', 'Enrolled', 'Graduated'], tickangle=0)

            st.plotly_chart(fig_semesters)
        
        except ValueError as e:
            st.write("No data available to display.")

    #------------------------
    container = st.container(border=True)
    col1, col2 = st.columns([4,1])
    with container:
        with col1:
            # st.markdown("<h3 style='text-align: center;'>Dropout Rate by Course</h3>", unsafe_allow_html=True)
            st.subheader("Dropout Rate by Course")

            course_kls = kelas

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

            course_kls['Course'] = course_kls['Course'].map(category_mapping)

            data_do = course_kls[course_kls['Status_0'] == 1]
            data_notdo = course_kls.loc[course_kls['Status']>0]

            course_do = data_do.groupby('Course')['Status_0'].sum()
            course_notdo = data_notdo.groupby('Course')['Status_New'].sum()

            try:
                total_course = round((course_do / (course_do + course_notdo) * 100), 2)
                a_sorted = total_course.sort_values()
                maxA = a_sorted.idxmax()

                fig = px.bar(
                    x=a_sorted.values, 
                    y=a_sorted.index,
                    labels={'y': 'Course', 'x': 'Dropout Rate (%)'},
                    text=[f"{value}%" for value in a_sorted.values],
                    color=a_sorted.index,
                    color_discrete_sequence=['#393939' if b != maxA else 'white' for b in a_sorted.index],
                    height=600
                )
                
                fig.update_traces(
                    textposition='outside',
                )

                fig.update_layout(
                    plot_bgcolor='#111111',
                    paper_bgcolor='#111111',
                    font=dict(color='white'),
                    xaxis=dict(tickfont=dict(color='white'), linecolor='white'),
                    yaxis=dict(tickfont=dict(color='white'), linecolor='white'),
                    showlegend=False
                )

                st.plotly_chart(fig)

            except ValueError as e:
                st.write("No data available to display.")

        #-----
        with col2:
            create_pie_chart('Educational_special_needs', 'Educational Special Needs <br>Distribution')
            create_pie_chart('Debtor', 'Debtor Distribution')
            create_pie_chart('Tuition_fees_up_to_date', 'Tuition Fees Up to Date<br>Distribution')
        #---------------------------------------------
    container = st.container(border=True)
    colors = ['white', '#393939']
    with container:
        colA, colB = st.columns([4,1])
        with colA:
            try:
                fig = px.histogram(
                    kelas,
                        x='Age_at_enrollment',
                        title='Age at Enrollment Distribution',
                        color_discrete_sequence=colors
                    )
                
                fig.update_layout(
                title=dict(
                    text='Age at Enrollment Distribution',
                    font=dict(
                        size=24
                        )
                    )
                )

                st.plotly_chart(fig)

            except ValueError as e:
                st.write("No data available to display.")

        with colB:
            max_age = (kelas['Age_at_enrollment'].max())
            mean_age =  (kelas['Age_at_enrollment'].mean())
            min_age = (kelas['Age_at_enrollment'].min())
            st.markdown(add_card(f"<b>Minimum age</b><br>{min_age}"), unsafe_allow_html=True)
            st.markdown(add_card(f"<b>Average age</b><br>{mean_age}"), unsafe_allow_html=True)
            st.markdown(add_card(f"<b>Maximum age</b><br>{max_age}"), unsafe_allow_html=True)

if add_selectbox == "Prediction":
    st.subheader("Prediction")
    course_list = list(data.Course_Label.unique())[::-1]
    course_list.sort()

    if 'pred_selected' not in st.session_state:
        st.session_state.pred_selected = None

    if st.session_state.pred_selected is None:
        course_selected = st.selectbox('Course', ['None', *course_list])
    else:
        course_selected = st.selectbox('Course', course_list)

    if course_selected == 'None':
        st.error("Please select a valid course.")
    else:
        st.session_state.course_selected = course_selected

    reverse_mapping = {v: k for k, v in category_mapping.items()}

    if course_selected != 'None':
        course_selected = reverse_mapping[course_selected]

    # ===============================================================

    if course_selected in [9991, 8014]:
        time_selected=0
    else:
        time_selected=1

    # ===============================================================

    admgrade_selected = st.number_input("Admission grade", value=0.0, step=0.1, min_value=0.0, max_value=200.0)
    admgrade_selected = round(admgrade_selected,1)

    # ===============================================================

    colGender, colAge = st.columns(2)

    with colGender:
        gender_list = ['Male', 'Female']
        gender_selected = st.selectbox('Gender', (gender_list))

        if gender_selected=="Female":
            gender_selected=0
        elif gender_selected=="Male":
            gender_selected=1

    with colAge:
        age_selected = st.number_input("Age at enrollment", step=1, min_value=17, max_value=70)

    # ===============================================================
    bool1, bool2 = st.columns(2)

    with bool1:
        special_list = ['Yes', 'No']
        special_selected = st.radio('Special education needs?', (special_list))

        if (special_selected=="No"):
            special_selected=0
        elif(special_selected=="Yes"):
            special_selected=1
        
    # ===============================================================
    with bool2:
        debtor_list = ['Yes', 'No']
        debtor_selected = st.radio('Debtor?', (debtor_list))

        if (debtor_selected=="No"):
            debtor_selected=0
        elif(debtor_selected=="Yes"):
            debtor_selected=1

    # ===============================================================
    bool3, bool4 = st.columns(2)
    with bool3:
        tuition_list = ['Yes', 'No']
        tuition_selected = st.radio('Tuition up to date?', (tuition_list))

        if (tuition_selected=="No"):
            tuition_selected=0
        elif(tuition_selected=="Yes"):
            tuition_selected=1

    # ===============================================================
    with bool4:
        scholarship_list = ['Yes', 'No']
        scholarship_selected = st.radio('Scholarship holder?', (scholarship_list))

        if (scholarship_selected=="No"):
            scholarship_selected=0
        elif(scholarship_selected=="Yes"):
            scholarship_selected=1

    # ===============================================================
    grade1, grade2 = st.columns(2)

    with grade1:   
        grade1_selected = st.number_input("First semester grade", value=0.0, step=0.1, min_value=0.0, max_value=20.0)
        grade1_selected = round(grade1_selected,2)
    
    with grade2:
        grade2_selected = st.number_input("Second semester grade", value=0.0, step=0.1, min_value=0.0, max_value=20.0)
        grade2_selected = round(grade2_selected,2)

    st.markdown('<style>div.stButton > button {margin: 0 auto; display: block; background: white; color: black;}</style>', unsafe_allow_html=True)
    button_predict = st.button("Predict", key='custom_button')
    if button_predict:
        if course_selected=="None":
            st.write("Please select a valid course.")
        else:
            model = load('model/model.joblib')
            user_data = {
                'Course': [course_selected], 
                'Daytime_evening_attendance': [time_selected], 
                'Admission_grade': [admgrade_selected], 
                'Educational_special_needs': [special_selected], 
                'Debtor': [debtor_selected], 
                'Tuition_fees_up_to_date': [tuition_selected], 
                'Gender': [gender_selected], 
                'Scholarship_holder': [scholarship_selected], 
                'Age_at_enrollment': [age_selected], 
                'Curricular_units_1st_sem_grade': [grade1_selected],
                'Curricular_units_2nd_sem_grade': [grade2_selected]
            }

            X_new = pd.DataFrame(user_data)
            predictions = model.predict(X_new)
            st.subheader("Prediction Result")
            if predictions == 0:
                st.write("Student is likely to dropout.")
            elif predictions == 1:
                st.write("Student is NOT likely to dropout.")
