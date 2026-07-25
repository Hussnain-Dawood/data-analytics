import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load original file (raw, no date parsing)
# ═══════════════════════════════════════════════════════════════════════════
df = pd.read_excel(
    'ICT701 Assignment1_RetailStore_Dataset.xlsx',
    sheet_name='RetailStore Dataset ',
    dtype={'Date': object}
)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Fix dates
# Problem: Excel stored some dates as datetime objects with day/month swapped
#          e.g. datetime(2022,5,1) actually means dd=5, mm=1 → Jan 5
#          String dates like '1/27/2022' are already correct (m/d/yyyy)
# ═══════════════════════════════════════════════════════════════════════════
def fix_date(val):
    if isinstance(val, datetime.datetime):
        try:
            return datetime.date(val.year, val.day, val.month)   # swap day<->month
        except:
            return val.date()
    elif isinstance(val, str):
        return pd.to_datetime(val, dayfirst=False).date()
    return val

df['Date'] = pd.to_datetime(df['Date'].apply(fix_date), errors='coerce')

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Format Time column
# ═══════════════════════════════════════════════════════════════════════════
def format_time(val):
    if isinstance(val, datetime.time):
        return val.strftime('%H:%M')
    try:
        return pd.to_datetime(str(val)).strftime('%H:%M')
    except:
        return str(val)

df['Time'] = df['Time'].apply(format_time)

df = df.sort_values('Date').reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Summary stats
# ═══════════════════════════════════════════════════════════════════════════
print("Shape:", df.shape)
print("Date range:", df['Date'].min().strftime('%m/%d/%Y'), "->", df['Date'].max().strftime('%m/%d/%Y'))
print("Months present:", sorted(df['Date'].dt.month.unique()))   # must be [1, 2, 3]
print("Total Revenue:      $", round(df['Total'].sum(), 2))
print("Total Gross Income: $", round(df['gross income'].sum(), 2))
print("Avg Transaction:    $", round(df['Total'].mean(), 2))
print("Avg Rating:          ", round(df['Rating'].mean(), 2))

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Branch Revenue + Payment Method
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

branch_sales = df.groupby('Branch')['Total'].sum().reset_index()
colors = ['#1565C0', '#C62828', '#2E7D32']
bars = axes[0].bar(branch_sales['Branch'], branch_sales['Total'],
                   color=colors, edgecolor='white', width=0.5)
axes[0].set_title('Total Sales Revenue by Branch', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Branch')
axes[0].set_ylabel('Total Revenue ($)')
for bar, val in zip(bars, branch_sales['Total']):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+500,
                 f'${val:,.0f}', ha='center', fontsize=9)
axes[0].set_ylim(0, branch_sales['Total'].max()*1.18)
axes[0].tick_params(bottom=False)

payment_counts = df['Payment'].value_counts()
axes[1].pie(payment_counts.values, labels=payment_counts.index, autopct='%1.1f%%',
            colors=['#42A5F5','#66BB6A','#FFA726'], startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':1.5})
axes[1].set_title('Payment Method Distribution', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig1_branch_payment.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig1 saved")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Product Line Revenue + Rating
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

pl_rev = df.groupby('Product line')['Total'].sum().sort_values(ascending=True)
pl_colors = ['#EF9A9A','#F48FB1','#CE93D8','#9FA8DA','#80CBC4','#A5D6A7']
bars = axes[0].barh(pl_rev.index, pl_rev.values, color=pl_colors, edgecolor='white')
axes[0].set_title('Revenue by Product Line', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Total Revenue ($)')
for bar, val in zip(bars, pl_rev.values):
    axes[0].text(val+300, bar.get_y()+bar.get_height()/2,
                 f'${val:,.0f}', va='center', fontsize=8)
axes[0].set_xlim(0, pl_rev.max()*1.18)

pl_rating = df.groupby('Product line')['Rating'].mean().sort_values(ascending=False)
axes[1].bar(range(len(pl_rating)), pl_rating.values,
            color='#5C6BC0', edgecolor='white', width=0.6)
axes[1].set_xticks(range(len(pl_rating)))
axes[1].set_xticklabels([l[:14]+'.' if len(l)>14 else l for l in pl_rating.index],
                        rotation=30, ha='right', fontsize=8)
axes[1].set_title('Average Customer Rating by Product Line', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Average Rating')
axes[1].set_ylim(0, 10)
avg = df['Rating'].mean()
axes[1].axhline(y=avg, color='red', linestyle='--', alpha=0.7, label=f'Overall Avg: {avg:.2f}')
axes[1].legend()
plt.tight_layout()
plt.savefig('fig2_product.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig2 saved")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Monthly Sales Trend + Customer Type
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

monthly = df.groupby([df['Date'].dt.to_period('M'), 'Branch'])['Total'].sum().unstack(fill_value=0)
monthly.index = monthly.index.astype(str)
for branch, color in zip(['X','Y','Z'], ['#1565C0','#C62828','#2E7D32']):
    if branch in monthly.columns:
        axes[0].plot(monthly.index, monthly[branch], marker='o',
                     label=f'Branch {branch}', color=color, linewidth=2)
axes[0].set_title('Monthly Sales Trend by Branch (Jan-Mar 2022)',
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Total Revenue ($)')
axes[0].legend()
axes[0].tick_params(axis='x', rotation=15)

ct_pl = df.groupby(['Customer type','Product line'])['Total'].sum().unstack(fill_value=0)
ct_pl.T.plot(kind='bar', ax=axes[1], color=['#42A5F5','#FFA726'],
             edgecolor='white', width=0.7)
axes[1].set_title('Product Line Revenue by Customer Type', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Product Line')
axes[1].set_ylabel('Total Revenue ($)')
axes[1].set_xticklabels([l[:12]+'.' if len(l)>12 else l for l in ct_pl.columns],
                        rotation=30, ha='right', fontsize=8)
axes[1].legend(title='Customer Type')
plt.tight_layout()
plt.savefig('fig3_trend_customer.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig3 saved")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Correlation Matrix + Gender vs Product Line
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

num_cols = ['Unit price','Quantity','Tax 5%','Total','cogs','gross income','Rating']
corr = df[num_cols].corr()
sns.heatmap(corr, ax=axes[0], annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, linewidths=0.5, annot_kws={'size':8})
axes[0].set_title('Correlation Matrix - Numerical Variables', fontsize=13, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45, labelsize=8)

gender_pl = df.groupby(['Gender','Product line'])['Total'].sum().unstack(fill_value=0)
gender_pl.T.plot(kind='bar', ax=axes[1], color=['#E91E63','#1976D2'],
                 edgecolor='white', width=0.7)
axes[1].set_title('Product Line Revenue by Gender', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Product Line')
axes[1].set_ylabel('Total Revenue ($)')
axes[1].set_xticklabels([l[:12]+'.' if len(l)>12 else l for l in gender_pl.columns],
                        rotation=30, ha='right', fontsize=8)
axes[1].legend(title='Gender')
plt.tight_layout()
plt.savefig('fig4_corr_gender.png', dpi=150, bbox_inches='tight')
plt.close()
print("fig4 saved")

# ═══════════════════════════════════════════════════════════════════════════
# PRINT ALL STATS FOR REPORT
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== BRANCH REVENUE ===")
print(df.groupby('Branch')['Total'].sum().round(2))
print("\n=== PRODUCT LINE REVENUE ===")
print(df.groupby('Product line')['Total'].sum().sort_values(ascending=False).round(2))
print("\n=== PAYMENT % ===")
print((df['Payment'].value_counts(normalize=True)*100).round(1))
print("\n=== CUSTOMER TYPE ===")
print(df['Customer type'].value_counts())
print("\n=== RATING BY BRANCH ===")
print(df.groupby('Branch')['Rating'].mean().round(2))
print("\n=== RATING BY PRODUCT LINE ===")
print(df.groupby('Product line')['Rating'].mean().round(2))
print("\n=== MONTHLY REVENUE ===")
print(df.groupby(df['Date'].dt.to_period('M'))['Total'].sum().round(2))
print("\n=== DESCRIBE ===")
print(df[['Unit price','Quantity','Total','gross income','Rating']].describe().round(2).to_string())
print("\nAll figures saved.")
