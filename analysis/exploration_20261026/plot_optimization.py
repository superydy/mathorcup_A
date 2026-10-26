from pathlib import Path
import sys,json
W=Path(__file__).resolve().parent
sys.path.insert(0,str(W.parent/'paper_v3_revision/runtime_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
q=json.loads((W/'q1_summary.json').read_text(encoding='utf-8'));i=json.loads((W/'q1_independent_summary.json').read_text(encoding='utf-8'));p=json.loads((W/'q1_posthoc_blend_summary.json').read_text(encoding='utf-8'));g=json.loads((W/'q3_shared_cov_summary.json').read_text(encoding='utf-8'))['results']
fig,(a,b)=plt.subplots(1,2,figsize=(12,5))
pts=[('原ET',q['ET_original'],'#6e6e6e',(-5,-22)),('共享与独立ET等权',i['ET_half_independent'],'#168b72',(-70,10)),('内层选择后保守融合',q['inner_half_blend'],'#3875ca',(5,-18)),('L1提升树',i['LGBM_L1'],'#cc7831',(-10,-20)),('ET/L1融合（事后候选）',p,'#9b59b6',(-100,8))]
for label,r,c,offset in pts:
    a.scatter(r['MAE'],r['R2'],s=65,c=c,zorder=3);a.annotate(label,(r['MAE'],r['R2']),xytext=offset,textcoords='offset points',fontsize=9,color=c)
a.set_xlabel('宏平均MAE（越小越好）');a.set_ylabel('宏平均R²（越大越好）');a.set_xlim(.330,.348);a.set_ylim(.198,.274);a.grid(alpha=.2);a.set_title('Q1 新增探索与原基线')
for label,key,c,offset in [('原潜在高斯','common_original','#6e6e6e',(12,-5)),('共享相关混合','common_shared_mix','#168b72',(-12,10)),('完全共享相关','common_shared_only','#3875ca',(-100,-18))]:
    r=g[key];b.scatter(r['Wasserstein'],r['correlation_RMSE'],s=65,c=c,zorder=3);b.annotate(label,(r['Wasserstein'],r['correlation_RMSE']),xytext=offset,textcoords='offset points',fontsize=9,color=c)
b.set_xlabel('Wasserstein边际距离（越小越好）');b.set_ylabel('相关矩阵RMSE（越小越好）');b.set_xlim(.9635,.9705);b.set_ylim(.298,.320);b.grid(alpha=.2);b.set_title('Q3 保留边际的相关结构优化')
fig.text(.5,.015,'均为本数据上的新增探索；重复使用已有划分。Q3按3折与5个采样种子汇总，采样种子不增加真实验证样本。',ha='center',fontsize=9,color='#666666')
fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(W/'优化对比.png',dpi=180,bbox_inches='tight');plt.close(fig)
