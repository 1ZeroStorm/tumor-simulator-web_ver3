import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# Import environment lokal Anda
from environment_v3 import CancerSimulation  
from analyzer_v3 import PatientAnalyzer

# --- CUSTOM CALLBACK UNTUK STREAMLIT ---
class StreamlitLogCallback(BaseCallback):
    def __init__(self, log_file="live_metrics.csv", save_freq=20, verbose=0):
        super().__init__(verbose)
        self.log_file = log_file
        self.save_freq = save_freq
        self.rewards = []
        self.data_history = []

    def _on_step(self) -> bool:
        # 1. Ambil info dari env (info dikirim sebagai list)
        infos = self.locals['infos'][0]
        
        # 2. Ambil data dasar
        step_data = {
            'Step': self.num_timesteps,
            'Reward': self.locals['rewards'][0],
            'Action': self.locals['actions'][0],
            'info': infos
        }
        
        self.data_history.append(step_data)


        # 3. Simpan ke CSV secara periodik
        if self.num_timesteps % self.save_freq == 0:
            df = pd.DataFrame(self.data_history)
            df.to_csv(self.log_file, index=False)
            
        return True

def main():
    save_path = "./local_models/"
    os.makedirs(save_path, exist_ok=True)
    print("training")

    analyzer = PatientAnalyzer("data/Gene_Expression_Analysis_and_Disease_Relationship_Synthetic.csv")
    profile = analyzer.get_strategic_profile()
    print(profile)
    env = CancerSimulation(profile=profile) 

    # Gabungkan dua callback: Satu untuk save model, satu untuk Streamlit
    checkpoint_cb = CheckpointCallback(save_freq=1000, save_path=save_path, name_prefix="oncosteer")
    streamlit_cb = StreamlitLogCallback(log_file="live_metrics.csv", save_freq=1000)

    print("Memulai local training...")
    model = PPO("MlpPolicy", env, verbose=0) # Matikan verbose agar terminal bersih
    
    # Masukkan callback ke dalam list
    model.learn(total_timesteps=100000, callback=[checkpoint_cb, streamlit_cb])

    model.save(os.path.join(save_path, "oncosteer_final"))
    print("Training Selesai!")

if __name__ == "__main__":
    main()