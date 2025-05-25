import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 1️⃣ 读取数据（使用你自己的本地路径）
energy_df = pd.read_csv(
    '/Users/viewyan/fairchem/src/fairchem/data/odac/promising_mof/promising_mof_energies/adsorption_energy.txt',
    sep=' ',
    header=None,
    names=['mof_id', 'adsorption_energy']
)

features_df = pd.read_csv(
    '/Users/viewyan/fairchem/src/fairchem/data/odac/promising_mof/promising_mof_features/output.txt',
    sep='\t'
)

# 2️⃣ 清洗 & 合并
features_df['Material'] = features_df['Material'].str.replace('.cif', '', regex=False)
energy_df['Material'] = energy_df['mof_id'].apply(lambda x: x.split('_w_')[0])
merged_df = pd.merge(energy_df, features_df, on='Material', how='inner')
print(f"✅ Matched samples: {len(merged_df)}")

# 3️⃣ 模型训练
X = merged_df[['M_O_M', 'Benzene', 'Parellel', 'Parellel68',
               'DistanceParallel68', 'DistanceParallel68Direct', 'uncoordN']]
y = merged_df['adsorption_energy']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"🌟 MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"🌟 R² Score: {r2_score(y_test, y_pred):.4f}")

# 4️⃣ 可视化 - 对比全体 vs 建模数据的吸附能分布
plt.figure(figsize=(8, 5))
plt.hist(energy_df['adsorption_energy'], bins=30, alpha=0.5,
         label='All MOFs', color='lightgray', edgecolor='black')
plt.hist(merged_df['adsorption_energy'], bins=30, alpha=0.7,
         label='Used for model', color='skyblue', edgecolor='black')
plt.xlabel("CO₂ Adsorption Energy (eV)", fontsize=12)
plt.ylabel("Number of MOFs", fontsize=12)
plt.title("CO₂ Adsorption Energy Distribution", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
