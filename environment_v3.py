import gymnasium as gym
import numpy as np

class CancerSimulation(gym.Env):
    def __init__(self, profile):
        super(CancerSimulation, self).__init__()
        self.profile = profile

        self.observation_space = gym.spaces.Box(low=0, high=1e6, shape=(4,), dtype=np.float32)
        # defines 1D array with 4 elements (ex: [speed, distance, temperature, toxicity])
        # each elements can't surpass 1 million
        # lowest value is 0 (can't be negative)
        # float 32 bit for standard neural network 

        self.action_space = gym.spaces.Discrete(3)
        # 3 fixed moves (0, 1, 2) (ex: 0: left, 1: straight, 2: right) 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # takes the seed you provided and re-seeds self.np_random
        # seed = None -> generates a random seed for you

        # State: [Size, Res_A, Res_B, Toxicity]
        self.state = np.array([self.profile['normalized_init_tumor_size'], self.profile['starting_res_a'], self.profile['starting_res_b'], 0.0], dtype=np.float32)
        # 1000 -> inital tumor size
        # initial resistance for drug A
        # 5.0 -> initial resistance for drug B
        # 0.0 -> initial toxicity

        self.day = 0
        self.consecutive_drugs = 0
        
        return self.state, {}

    def step(self, action):
        normalized_size_before, res_a, res_b, toxicity = self.state
        size_before = normalized_size_before * self.profile['initial_tumor_size']
        self.day += 1
        res_b_before = res_b # Simpan nilai res_b sebelum update untuk perhitungan reward "Trap" nanti
        kill_rate = 0

        # 1. Natural Tumor Growth
        size = size_before * (1.0 + self.profile['avg_growth'] / 100.0) # if avg_growth is 14, it means the tumor grows by 14% each day.
        size_after_duplication = size # prior to drug

        # 2. Toxicity Logic (Keep your existing constraints) - either drug A or B, it adds toxicity and consecutive drug counter
        if action == 1 or action == 2:
            toxicity += self.profile['tox_increment']
        else: # if none actions are taken, toxicity reduced 
            
            toxicity = max(0.0, toxicity - (self.profile['tox_increment'] * 0.5))

        # 3. Drug Effects
        if action == 1: # Drug A (Standard)
            kill_rate = max(0.05, 0.9 - (res_a / self.profile['max_res_a'])) # kill rate minimum of  0.05, 0.9 is Theoretical Ceiling dari sebuah obat
            size -= (size * kill_rate) # 5% of size is reduced (bare minimum)
            res_a += 0.3
            res_b -= 1 #self.profile['trap_sensitivity'] # The Trap: Making the cancer weak to Drug B
        
        elif action == 2: # Drug B (The Trap) 
            kill_rate = 0.85 / (1 + (res_b / 2)**4) # kill rate high if res b is below 2.5, else kill rate low
            size -= (size * kill_rate) 
            res_b += 0.2
            
            
        
        # 4. NEW REWARD LOGIC
        reward = 0
        
        # A. Proportional Shrinkage (Perkuat pengali untuk tumor kecil)
        shrinkage_pct = (size_before - size) / self.profile['initial_tumor_size']
        
        if shrinkage_pct > 0:
            # Beri hadiah besar jika berhasil mengecilkan tumor
            reward += (shrinkage_pct * 500.0) 
        else:
            # Penalti Pertumbuhan: Dibuat EKSPONENSIAL
            growth_pct = abs(shrinkage_pct)
            # Jika tumor tumbuh, hukum sangat berat agar AI takut 'Rest' terlalu lama
            reward -= (growth_pct * 600.0) 
            
            # Tambahan: Jika ukuran tumor melewati batas awal, hukum ekstra
            if size > self.profile['initial_tumor_size']:
                reward -= 20.0

        # B. Toxicity Penalties (Beri "Safe Zone")
        # Jangan hukum AI jika toksisitas masih rendah (misal < 3.0)
        if toxicity > 4.0:
            if toxicity > 8.0:
                reward -= (toxicity ** 2.5) # Sangat bahaya
            else:
                reward -= (toxicity * 1.2) # Ringan di zona aman

        # C. Strategic "Trap" Bonus (Sangat Bagus, Pertahankan)
        res_b_drop = res_b_before - res_b
        if res_b_drop > 0 and action == 1:
            reward += (res_b_drop * 20.0) # Naikkan agar AI semangat "set up the trap"

        # D. TIME PENALTY (PENTING!)
        # Hukum AI setiap hari yang berlalu agar ia buru-buru menyelesaikan simulasi
        reward -= (self.day * 2)

        

        # --- PERBAIKAN 3: STATE NORMALIZATION ---
        # AI (Neural Network) "buta" jika melihat angka 1000 (size) berdampingan dengan angka 2 (res_b).
        # Kita harus menormalkan size agar skalanya mirip dengan variabel lain (0.0 - 1.0).
        self.state = np.array([size/self.profile['initial_tumor_size'], res_a, res_b, toxicity], dtype=np.float32)

        if size <= 2: # Jika sisa tumor kurang dari setengah sel
            size = 0
        # 5. Termination Logic
        done = bool(size <= 0 or self.day >= 50 or toxicity > 10.0)
        
        # Di dalam env.py bagian reward
        if action == 2: # Saat menggunakan Drug B
            if res_b < 2:
                # Bonus besar jika menggunakan Drug B di saat resistensinya rendah (hasil kerja Drug A)
                reward += 500
            
            # Berikan reward tambahan jika kill_rate yang dihasilkan Drug B sangat tinggi
            if kill_rate > 0.7:
                reward += 50.0
        
        

        if size < 1: 
            reward += 2000 # Jackpot diraih
        elif toxicity > 10.0:
            reward -= 1000 # Hukuman mati
        elif self.day >= 30:
            reward -= 500 # Kehabisan waktu
            
        info = {
            'tumor_size_before':size_before,
            'size_after_duplication': size_after_duplication,
            'size_after_drug':size,
            'day': self.day,
            'action taken': action,
            'kill rate': kill_rate,
            'res b': res_b,
            'res a': res_a
        }
            
        return self.state, reward, done, False, info