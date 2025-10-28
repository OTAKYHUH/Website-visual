from build_data import get_tables as data_tables
from build_H import get_tables as h_tables
from build_safety import get_tables as safety_tables
from photos import get_tables as staff_tables

dfs_data = data_tables()                            # p123_long, p123_enriched, p456_enriched, appended, etc.
dfs_h    = h_tables()                               # Hourly, Log, Details (+ raw variants)
dfs_safe = safety_tables()                          # Demand, Demand_2, Safety, Terminal
#dfs_pic  = staff_tables(r"C:\Users\ahgua\PSA International\(PSAC-CNBD)-YOD-efile - Cess\Staff Photo")  # photos_df, photos_dict
dfs_pic  = staff_tables(r"C:\Users\ahgua\Documents\Website-visual\static\Staff Photo")
print(dfs_data["p123_enriched"])
demand   = dfs_safe["Demand"]
p123     = dfs_data.get("p123_enriched")
