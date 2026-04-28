import streamlit as st
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from analyzer_v3 import PatientAnalyzer
from environment_v3 import CancerSimulation
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

# --- HELPER FUNCTION: TUMOR VISUALIZATION ---
def create_tumor_visualization(tumor_size, resistance_list, max_res=15.0, extratitle=None, chart_key=None): 
    """
    High-performance tumor cell visualization using drawable canvas.
    Maps the AVERAGE resistance value to the color of all cells.
    """
    if 'cell_coordinates' not in st.session_state:
        # Persistent coordinate pool to keep cell positions stable during refresh
        # 20000 rows and 2 cols (represents coords interval 0 to 1) 
        st.session_state.cell_coordinates = np.random.rand(20000, 2)

    num_cells = int(min(len(st.session_state.cell_coordinates), max(0, tumor_size))) 
    cell_coords = st.session_state.cell_coordinates[:num_cells].copy()

    # 1. Handle Resistance Data (Calculate Average)
    if isinstance(resistance_list, (list, np.ndarray)) and len(resistance_list) > 0: 
        # Mengambil nilai rata-rata dari seluruh list resistensi
        avg_res = np.mean(resistance_list)
    elif isinstance(resistance_list, (int, float)):
        # Jika argumen yang dimasukkan langsung berupa angka rata-rata tunggal
        avg_res = float(resistance_list)
    else:
        # Fallback if resistance_list is empty or invalid
        avg_res = 5.0 

    # 2. Normalize the average for colorscale (0 to 1)
    avg_norm = np.clip(avg_res / max_res, 0, 1)
    
    # Membuat array di mana SEMUA sel memiliki nilai warna rata-rata yang sama
    norm_colors = np.full(num_cells, avg_norm)

    # 3. Create Plotly scatter plot with colorbar
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cell_coords[:, 0],
        y=cell_coords[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=norm_colors,
            colorscale='YlOrRd',
            # PENTING: Kunci batas bawah dan atas agar colorbar tidak rusak
            cmin=0.0, 
            cmax=1.0, 
            showscale=True,
            colorbar=dict(
                title='Average<br>Resistance',
                thickness=15,
                len=0.7,
                x=1.02,
                # Opsional: Memperjelas angka di colorbar agar sesuai skala max_res
                tickvals=[0, 0.5, 1],
                ticktext=['0', f'{max_res/2:.1f}', f'{max_res:.1f}'] 
            ),
            opacity=0.8,
            line=dict(width=0)
        ),
        hoverinfo='skip',
        showlegend=False
    ))
    
    fig.update_layout(
        title=dict(
            text=f"{extratitle}<br><sup>Live Population: {num_cells:,} cells | Avg Res: {avg_res:.2f}</sup>",
            pad=dict(b=10),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        plot_bgcolor='#161B22',
        paper_bgcolor='#0E1117',
        font=dict(color='#888888', size=10),
        autosize=False,
        width=500,
        height=500,
        margin=dict(l=10, r=10, t=70, b=20),
        coloraxis_colorbar=dict(x=1.0, xanchor='left')
    )
    
    st.plotly_chart(fig, use_container_width=False, key=chart_key)


# --- CONFIGURATION ---
st.set_page_config( 
    page_title="OncoSteer: Evolutionary AI", # your browser bar on top
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="auto"
)

MODEL_PATH = "oncosteer_100000_steps" 
DEFAULT_DATA_PATH = "data/Gene_Expression_Analysis_and_Disease_Relationship_Synthetic.csv"
TESTING_ONE_DATA_PATH = "data/Replicated_Gene_Expression_Analysis.csv"
TESTING_TWO_DATA_PATH = "data/Synthetic_Replicated_Data.csv"

@st.cache_resource # decorate function load_model() to just run once even when the web refreshes
def load_model(): # returns the PPO model
    if os.path.exists(f"{MODEL_PATH}.zip"):
        #st.write("model found, loading...")
        return PPO.load(MODEL_PATH)
    return None

st.title("🧬 OncoSteer")
st.markdown("### Steering Tumor Evolution Toward Therapeutic Vulnerability.")

states = {
    'treatment_history': [],
    'generation_done': False,
    'cell_res_data': [],
    'source': 'empty',
    'data_mode' : None
}

# Gunakan sintaks dictionary untuk inisialisasi awal
for key, default_value in states.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- SIDEBAR #2 ---

st.sidebar.header("About OncoSteer")
sidebar_info_list = ["What is oncosteer?", "Powered by Azure Machine Learning", "Get Started!"]
sidebar_info_mode = st.sidebar.radio("Select Data Source:", sidebar_info_list)

# --- SIDEBAR ---

data_source = ["Upload CSV", "Use Default Synthetic Data - training data (kaggle)", "Testing data 1 (kaggle)", "Testing data 2 (kaggle)"]

if sidebar_info_mode == sidebar_info_list[2]:
    st.sidebar.header("Patient Data Input")

    st.session_state.data_mode = st.sidebar.radio("Select Data Source:", data_source)
else:
    st.session_state.data_mode = None

    if sidebar_info_mode == sidebar_info_list[0]:
        st.markdown("##### What is OncoSteer?")
        st.markdown("""
            ### 🧬 Understanding Genomic Input Data

            Users can access the website and upload a **CSV file** containing genomic data. This dataset serves as the foundational "environment" for our Reinforcement Learning agent, representing the unique biological profile of each patient.

            The CSV file must contain the following columns, which map directly to the simulation parameters:

            * **`Disease_Status`**: Indicates whether the sample is **Tumor** or **Healthy_Control**. This helps the system distinguish between the baseline health of the patient and the active malignancy.
            * **`Gene_A_Oncogene` (Tumor Proliferation Engine)**: 
                Represents the core driver of the cancer. Higher expression of this gene dictates both the **Initial Tumor Size** and the **Average Growth Rate**. It acts as the "motor" that the AI must counteract; the more aggressive this gene, the faster the tumor recovers between treatments.
            * **`Gene_D_Therapy` (Baseline Drug A Resistance)**: 
                Defines the starting point for resistance toward the primary therapy (**Drug A**). It represents the patient's innate genetic predisposition to resist standard treatment even before the first dose is administered.
            * **`Gene_B_Immune` (Dynamic Resistance to Drug B)**: 
                Used as the starting point for resistance toward the secondary "Trap" drug (**Drug B**). This maps how the immune-related genetic profile shields the tumor from targeted attacks. The AI's objective is to use Drug A to "break" this barrier, lowering the resistance derived from this gene to a critical threshold.
            * **`Gene_C_Stromal` (Toxicity Sensitivity)**: 
                Indicates the sensitivity of the surrounding healthy stromal tissue. In our model, this determines the **Toxicity Increment** per dose. A high value means the patient’s body is more vulnerable to the side effects of the drugs, requiring the AI to be more cautious to avoid reaching the lethal toxicity limit.

            ---

            ### 📊 Simulation & Testing
            Users can also access **Pre-loaded Test Data** available on the dashboard. This allows for real-time demonstrations using diverse patient profiles that were not included in the original training dataset, proving the **generalizability** and **robustness** of the PPO agent in handling unseen genomic signatures.
            """)

    elif sidebar_info_mode == sidebar_info_list[1]:
        st.markdown("##### Leveraging Azure ML for advanced AI training")
        st.markdown(
            "The developer uses Python as the main programming language, along with several external libraries such as Streamlit for web development, Pandas and NumPy for data management, and Stable-Baselines3 for implementing the PPO algorithm. \
            In addition, the developer utilizes several Azure platform services, including:  \n"
            "- **Azure Machine Learning (AML):** used to train models, monitor metrics, and store models in the Registry.  \n"
            "- **Azure Compute:** provides virtual machines (VMs) and development environments such as Visual Studio.  \n"
            "- **Azure Blob Storage:** serves as storage for trained models in `.zip` format."
        )




uploaded_file = None
current_data_source = None


#input csv, stored in uploaded_file

if  st.session_state.data_mode == data_source[0]:
    uploaded_file = st.sidebar.file_uploader("Upload Proteomic CSV", type=["csv"])
    if uploaded_file:
        current_data_source = uploaded_file.name
elif st.session_state.data_mode == data_source[1]:
    if os.path.exists(DEFAULT_DATA_PATH):
        uploaded_file = DEFAULT_DATA_PATH
        current_data_source = data_source[1]
    else:
        st.sidebar.error("file not found.")
elif st.session_state.data_mode == data_source[2]:
    if os.path.exists(TESTING_ONE_DATA_PATH):
        uploaded_file = TESTING_ONE_DATA_PATH
        current_data_source = data_source[2]
    else:
        st.sidebar.error("file not found.")
elif st.session_state.data_mode == data_source[3]:
    if os.path.exists(TESTING_TWO_DATA_PATH):
        uploaded_file = TESTING_TWO_DATA_PATH
        current_data_source = data_source[3]
    else:
        st.sidebar.error("file not found.")

# 1. Inisialisasi State (Cara yang benar)
# Jangan cek 'is None', Streamlit otomatis menyediakannya sebagai object.


# 2. Fungsi Reset
def reset_all():
    # Menggunakan atribut atau key sama saja, tapi konsisten itu penting
    st.session_state.generation_done = False
    st.session_state.treatment_history = []
    st.session_state.cell_res_data = []

# 3. Deteksi Perubahan Source
# Pastikan 'current_data_source' sudah terdefinisi (misal dari uploader)
if 'current_data_source' in locals() or 'current_data_source' in globals():
    if current_data_source != st.session_state.source:
        st.session_state.source = current_data_source
        reset_all()
        # Opsional: Paksa refresh agar UI bersih seketika
        # st.rerun()
    

# check if uploaded_file is not empty
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file) # pandas csv of data
    model = load_model() # loading the PPO model from azure
    

    try:
        analyzer = PatientAnalyzer(df=data) # analyzer tool, to do/return stuff from the csv
        profile = analyzer.get_patient_profile(data) # return avg resistance, avg growth rate
        # Fetch individual resistance levels for all tumor cells
        st.session_state.cell_res_data = analyzer.get_cell_resistance_data()  # returns 1D array list of 2000+ resistance levels of every cell   
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    if st.button("Generate Optimized Treatment Plan"):
        if model:
            env = CancerSimulation(profile)
            obs, _ = env.reset()
            history = []
            
            curr_sz_normalized, curr_ra, curr_rb = obs[0], obs[1], obs[2]
            # curr_sz -> current population
            # curr_ra -> current resistance A
            # curr_rb -> current resistance B

            for day in range(1, 50):
                # 1. Before State (Captures duplication/growth from previous day)
                history.append({
                    "Day": day,
                    "Status": "Before Drug (Prior Duplication)",
                    "Action": "—",
                    "Tumor Size": int(curr_sz_normalized * profile['initial_tumor_size']),
                    "Resist A": float(curr_ra),
                    "Resist B": float(curr_rb)
                })
                
                # 2. Apply Drug
                action, _ = model.predict(obs, deterministic=True)
                act_int = int(action.item())
                act_map = {0: "Rest", 1: "Drug A", 2: "Drug B"}
                
                obs, reward, terminated, truncated, info = env.step(act_int)
                
                # Get size after duplication from info dictionary
                sz_after_duplication = info['size_after_duplication']
                
                # Add history entry for After Duplication
                history.append({
                    "Day": day,
                    "Status": "Before Drug (Post Duplication)",
                    "Action": "—",
                    "Tumor Size": int(sz_after_duplication),
                    "Resist A": float(curr_ra),
                    "Resist B": float(curr_rb)
                })
                
                # Update states for "After" and next "Before"
                curr_sz_normalized, curr_ra, curr_rb = obs[0], obs[1], obs[2]
                
                history.append({
                    "Day": day,
                    "Status": f"After {act_map[act_int]}",
                    "Action": act_map[act_int],
                    "Tumor Size": int(info["size_after_drug"]),
                    "Resist A": float(curr_ra),
                    "Resist B": float(curr_rb)
                })
                
                if int(info["size_after_drug"]) <= 0 or terminated: break
            
            st.session_state.treatment_history = history
        else:
            st.error(f"model not found: {model}")
        

        

    if st.session_state.treatment_history:
        df_hist = pd.DataFrame(st.session_state.treatment_history)

        # --- VISUALIZATION SECTION ---
        
        #adding slider
        st.markdown("---")
        st.subheader("📊 Display of first 10 data")
        st.write(data.head(10))

        st.subheader("🔬 Microscopic Tumor Evolution")
        st.markdown(
            f"Initially, data is taken. The simulation uses ```Gene_A_Oncogene``` to measure the average growth, \
                uses ```Gene_D_Therapy``` to get the average resistance for the first drug, \
                uses ```Gene_C_Stromal``` to measure dynamic toxicity increment after drug and uses \
                ```Gene_B_Immune``` to meansure the average resistance for the second drug\n \
                You can view the {int(df_hist['Day'].max())} days of treatment history below.  \
            In this simulation, tumor cells initially duplicate, grow, and gradually develop resistance to both drugs. \
            The model illustrates the principle of collateral sensitivity: tumor cells adapt to the first drug \
            (shown by increasing resistance), while simultaneously reducing resistance to the other drug(s)."
        )

        day_to_show = st.slider("Select Day to Visualize (drag the slider)", 1, int(df_hist['Day'].max()), 1)
        
        # 
        day_data = df_hist[df_hist['Day'] == day_to_show]
        c_row = day_data[day_data['Status'] == "Before Drug (Prior Duplication)"].iloc[0]
        b_row = day_data[day_data['Status'] == "Before Drug (Post Duplication)"].iloc[0]
        a_row = day_data[day_data['Status'].str.contains("After")].iloc[0]
        
        # --- BARIS 1: V1 DAN V2 ---
        col1, col2 = st.columns(2)

        with col1:
            # Menggunakan Markdown dengan text-align center agar lurus dengan grafik
            st.markdown("<h4 style='text-align: center; color: #888888;'>Prior Duplication</h4>", unsafe_allow_html=True)
            create_tumor_visualization(c_row["Tumor Size"], st.session_state.cell_res_data, extratitle="", chart_key=f"c_row_{c_row['Day']}")

        with col2:
            st.markdown("<h4 style='text-align: center; color: #888888;'>Post Duplication</h4>", unsafe_allow_html=True)
            create_tumor_visualization(b_row["Tumor Size"], st.session_state.cell_res_data, extratitle="", chart_key=f"b_row_{b_row['Day']}")

        st.markdown("---")
        
        # Menggunakan kolom dengan rasio yang lebih ramping agar V3 besar tapi tetap terkunci di tengah
        _, col_v3, _ = st.columns([0.2, 0.6, 0.2]) 

        with col_v3:
            # Menggunakan <div> dengan flexbox untuk memastikan alignment absolut
            st.markdown(f"""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">
                    <h2 style="color: #2ECC71; margin-bottom: -10px;">Final Result: After {a_row['Action']}</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Panggil fungsi visualisasi
            create_tumor_visualization(a_row["Tumor Size"], st.session_state.cell_res_data, extratitle="", chart_key=f"a_row_{a_row['Day']}")
        
        a = df_hist[df_hist['Day'] == int(df_hist['Day'].max())]
        
        b = a[a['Status'].str.contains("After")].iloc[0]['Tumor Size']
        
        st.markdown(
            f"PPO (Proximal Policy Optimization) serves as the 'brain' of OncoSteer. \
            It transforms static clinical data into an adaptive, \
            dynamic personalized treatment trajectory, demonstrating how AI can navigate the delicate \
            trade-offs between drug efficacy and patient safety. The performance of the model is demonstrated \
            through the dynamic evolution of tumor size above and 2 charts below. At the last day, the model \
            manages to achieve a final tumor size of {b} cells."
        )

        # Formatting Resist A and B to 2 decimal places
        formatted_df = df_hist.copy()
        formatted_df["Resist A"] = formatted_df["Resist A"].map(lambda x: f"{x:.2f}")
        formatted_df["Resist B"] = formatted_df["Resist B"].map(lambda x: f"{x:.2f}")
        
        # Apply slight background color to separate Before/After visually
        def style_rows(row):
            if "Before" in row["Status"]:
                return ['background-color: #1b212c'] * len(row)
            return [''] * len(row)

        st.dataframe(formatted_df.style.apply(style_rows, axis=1), use_container_width=True, hide_index=True)

        # --- BAR CHART SECTION ---
        st.markdown("---")
        st.subheader("📈 Tumor Size Dynamics")
        fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0E1117')
        ax.set_facecolor('#161B22')
        
        days = df_hist[df_hist['Status'] == "Before Drug (Prior Duplication)"]['Day']
        b_vals = df_hist[df_hist['Status'] == "Before Drug (Prior Duplication)"]['Tumor Size']
        dup_vals = df_hist[df_hist['Status'] == "Before Drug (Post Duplication)"]['Tumor Size']
        a_vals = df_hist[df_hist['Status'].str.contains("After")]['Tumor Size']
        
        x = np.arange(len(days))
        ax.bar(x - 0.3, b_vals, 0.25, label='Before Duplication', color='#E74C3C', alpha=0.7)
        ax.bar(x - 0.05, dup_vals, 0.25, label='After Duplication', color='#F39C12', alpha=0.8)
        ax.bar(x + 0.2, a_vals, 0.25, label='After Drug', color='#2ECC71', alpha=0.9)
        
        # Labels and Axis Setup
        ax.set_xlabel('Timeline (Treatment Days)', color='#C9D1D9', fontsize=10)
        ax.set_ylabel('Cell Count (Tumor Size)', color='#C9D1D9', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(days, color='#C9D1D9')
        ax.tick_params(axis='y', labelcolor='#C9D1D9')
        
        # Removing spines for a cleaner look
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363D')

        ax.legend(facecolor='#161B22', edgecolor='#30363D', labelcolor='#C9D1D9')
        ax.grid(axis='y', alpha=0.1)
        
        plt.tight_layout()
        st.pyplot(fig)

else:
    st.info("Select 'Get Started' and use a data source in the sidebar to begin. See what type of files can be uploaded at 'What is oncosteer?'")