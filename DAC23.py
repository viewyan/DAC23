import pandas as pd
import matplotlib.pyplot as plt

# 读取 adsorption_energy.txt 文件
energy_df = pd.read_csv(
    '/Users/viewyan/fairchem/src/fairchem/data/odac/promising_mof/promising_mof_energies/adsorption_energy.txt',  # 路径请根据你电脑实际路径调整
    sep=' ',
    header=None,
    names=['mof_id', 'adsorption_energy']
)

# 查看前几行确认一下
print(energy_df.head())

# 画出吸附能的分布直方图
plt.hist(energy_df['adsorption_energy'], bins=50, color='lightcoral', edgecolor='black')
plt.xlabel('CO₂ Adsorption Energy (eV)')
plt.ylabel('Number of MOFs')
plt.title('Distribution of CO₂ Adsorption Energy (DAC2023)')
plt.grid(True)
plt.tight_layout()
plt.show()

features_df = pd.read_csv('/Users/viewyan/fairchem/src/fairchem/data/odac/promising_mof/promising_mof_features/output.txt', sep='\t')
energy_df = pd.read_csv('/Users/viewyan/fairchem/src/fairchem/data/odac/promising_mof/promising_mof_energies/adsorption_energy.txt',
                        sep=' ', header=None, names=['mof_id', 'adsorption_energy'])

# 统一字段用于匹配：去掉文件名后缀 .cif（如果有）
features_df['Material'] = features_df['Material'].str.replace('.cif', '', regex=False)

# 合并：用模糊匹配，energy 中的 mof_id 包含 Material
energy_df['Material'] = energy_df['mof_id'].apply(lambda x: x.split('_w_')[0])  # 只保留前缀匹配部分

# 合并数据（使用 Material 字段对齐）
merged_df = pd.merge(energy_df, features_df, on='Material', how='inner')

# 看看合并结果
print(merged_df.head())
print(f"Total number of matched samples：{len(merged_df)}")

# 保存训练用的合并数据（可选）
merged_df.to_csv('dac2023_training_data.csv', index=False)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 去掉无用列，只保留特征列作为 X
X = merged_df[['M_O_M', 'Benzene', 'Parellel', 'Parellel68',
               'DistanceParallel68', 'DistanceParallel68Direct', 'uncoordN']]
y = merged_df['adsorption_energy']

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测并评估
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"🌟 Mean Absolute Error (MAE): {mae:.4f}")
print(f"🌟 R² Score: {r2:.4f}")


energy_df = pd.read_csv(
    '/Users/viewyan/fairchem/src/fairchem/data/odac/promising_mof/promising_mof_energies/adsorption_energy.txt',
    sep=' ',                # 用空格分隔 ID 和能量
    header=None,            # 没有标题行
    names=['mof_id', 'adsorption_energy']  # 手动添加列名
)

plt.figure(figsize=(8, 5))  # 图像大小
plt.hist(energy_df['adsorption_energy'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel("CO₂ Adsorption Energy (eV)", fontsize=12)
plt.ylabel("Number of MOFs", fontsize=12)
plt.title("Distribution of CO₂ Adsorption Energies (DAC2023)", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

from matminer.featurizers.structure import DensityFeatures
from pymatgen.core import Structure
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.io.cif")


cif_folder = "/Users/viewyan/fairchem/src/fairchem/data/odac/promising_mof/promising_mof_structures/pristine"

# 准备结构
structures, ids = [], []
for f in os.listdir(cif_folder):
    if f.endswith(".cif"):
        try:
            s = Structure.from_file(os.path.join(cif_folder, f))
            structures.append(s)
            ids.append(f.replace(".cif", ""))
        except: pass

# 初始化特征器
featurizers = [
    DensityFeatures()
]

# 提取特征
data, columns = [], []
for sid, struct in zip(ids, structures):
    row = [sid]
    skip = False
    for f in featurizers:
        try:
            feats = f.featurize(struct)
            row += feats
            if not columns:
                columns += f.feature_labels()
        except:
            skip = True
            break
    if not skip:
        data.append(row)

df = pd.DataFrame(data, columns=["mof_id"] + columns)
df.to_csv("dac23_stable_fingerprints.csv", index=False)
print(f"✅ Total number of matched samples：{len(df)}")
df_energy = pd.read_csv(
    "/Users/viewyan/fairchem/src/fairchem/data/odac/promising_mof/promising_mof_energies/adsorption_energy.txt",
    sep="\s+",
    header=None,
    names=["mof_id", "adsorption_energy"]
)
print(df_energy.head())
# 提取结构名到新列
df_energy["structure"] = df_energy["mof_id"].str.extract(r"^([A-Z0-9]+)")

# 再 merge
df_all = pd.merge(df_energy, df, left_on="structure", right_on="mof_id")
print(f"✅ Total number of matched samples：{len(df_all)}")

plt.figure(figsize=(8, 6))
plt.scatter(df_all["packing fraction"], df_all["adsorption_energy"], alpha=0.5, color='teal')
plt.xlabel("Packing Fraction")
plt.ylabel("CO₂ Adsorption Energy (eV)")
plt.title("Packing Fraction vs Adsorption Energy (DAC2023)")
plt.grid(True)
plt.tight_layout()
plt.show()

df_clean = df_all[(df_all["adsorption_energy"] > -5) & (df_all["adsorption_energy"] < 5)]
plt.figure(figsize=(8, 6))
plt.scatter(df_clean["packing fraction"], df_clean["adsorption_energy"], alpha=0.5, color='tomato')
plt.xlabel("Packing Fraction")
plt.ylabel("CO₂ Adsorption Energy (eV)")
plt.title("Filtered Packing Fraction vs Adsorption Energy (DAC2023)")
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(df_all["density"], df_all["adsorption_energy"], alpha=0.5, color="green")
plt.xlabel("Density (g/cm³)")
plt.ylabel("CO₂ Adsorption Energy (eV)")
plt.title("Density vs Adsorption Energy (DAC2023)")
plt.grid(True)
plt.tight_layout()
plt.show()

df_filtered = df_all[(df_all["adsorption_energy"] >= -5) & (df_all["adsorption_energy"] <= 5)]
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
sns.regplot(
    data=df_filtered,
    x="density",
    y="adsorption_energy",
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "red"},
    color="green"
)

plt.xlabel("Density (g/cm³)")
plt.ylabel("CO₂ Adsorption Energy (eV)")
plt.title("Density vs Adsorption Energy with Regression Line (DAC2023)")
plt.tight_layout()
plt.grid(True)
plt.show()

df_filtered = df_all[(df_all["adsorption_energy"] >= -5) & (df_all["adsorption_energy"] <= 5)]

import seaborn as sns

plt.figure(figsize=(8, 6))
sns.regplot(
    data=df_filtered,
    x="vpa",
    y="adsorption_energy",
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "blue"},
    color="orange"
)

plt.xlabel("Void Porous Area (Å³)")
plt.ylabel("CO₂ Adsorption Energy (eV)")
plt.title("VPA vs Adsorption Energy with Regression Line (DAC2023)")
plt.tight_layout()
plt.grid(True)
plt.show()

df_filtered = df_all[(df_all["adsorption_energy"] >= -5) & (df_all["adsorption_energy"] <= 5)]
plt.figure(figsize=(8, 5))
plt.hist(df_filtered["adsorption_energy"], bins=50, color="skyblue", edgecolor="black")
plt.title("Distribution of CO₂ Adsorption Energies (DAC2023)")
plt.xlabel("CO₂ Adsorption Energy (eV)")
plt.ylabel("Values Count")
plt.grid(True)
plt.tight_layout()
plt.show()

