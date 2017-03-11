
# coding: utf-8

# # Her skriver du opp 
# # - hvordan du har resonnert for å finne parameterverdier, 
# # - hva slags antagelser du har gjort, 
# # - hvilke søkeord og databaser du har brukt, 
# # - hvilken kvalitet i henhold til evidenshierarki kildene dine har, og 
# # - foretar beregninger av sannsynligheter

# In[2]:

import cmath as c
import numpy as np
import pandas as pd
import scipy as sc
from scipy import stats


# # For ikke mutasjonsbærere:
# # - fra 'at risk' til 'ny brystkreft'
# Hovedressonement: tolker denne som tilbakefall (lokalt og fjernt). 
# 
# Antagleser om personer i 'at risk': har gjennomgått brystbevarende kirurgi, medikamentell behandling, og strålebehandling.
# 
# Databaser: PubMed
# 
# Dato: 11.01.17
# 
# Søkeord: (((((recurrence) AND breast conserving treatment) OR breast conserving therapy)) AND "meta analysis"[Publication Type]) AND "review"[Publication Type] 
# 
# Resultater: 36, 1 valgt
# 
# Type kilde: Meta-analyse av individ data i 17 randomiserte studier, dekker 10801 kvinner, publisert i Lancet
# 
# Link: https://www.ncbi.nlm.nih.gov/pubmed/22019144

# In[3]:

# Time: 10 years
# N = 7287 (vaktpostlympfeknute negative)
# N_BCT+RT = 3675 
# Event = any recurrence
# Reported risk = 10-y risk(probability)/person-years year 0-9
# i = 0: age < 40
# i = 1: age 40-49
# i = 2: age 50-59
# i = 3: age 60-69
# i = 4: age 70 +
n_0 = 189 
x_0 = 74

n_1 = 576
x_1 = 124

n_2 = 1093
x_2 = 155

n_3 = 1138
x_3 = 137

n_4 = 679
x_4 = 41
# Merk! Å regne ut ratene her med bare x/n blir ikke korrekt fordi folk har falt av langs studietiden til død...


# 10 year-prob any recurrence per person years, P_i:
def p():
    p0_t10 = 0.361
    p1_t10 = 0.208
    p2_t10 = 0.15
    p3_t10 = 0.142
    p4_t10 = 0.088
    return p0_t10, p1_t10, p2_t10, p3_t10, p4_t10


# In[4]:

p()


# In[5]:

def prp(prob, time): # prp = probability to rate to probability (going for prob(t = a) to prob(t=b))
    prob = prob
    time = time
    rate = -c.log(1 - prob)/time
    return rate
    


# In[13]:

p0_t1 = prp(p()[0], 10)
p1_t1 = prp(p()[1], 10)
p2_t1 = prp(p()[2], 10)
p3_t1 = prp(p()[3], 10)
p4_t1 = prp(p()[4], 10)


age_prob = pd.DataFrame(data = [p0_t1, p1_t1, p2_t1, p3_t1, p4_t1], columns= ['recurrence'])


# In[14]:

age_prob['age'] = ['<40', '40-49', '50-59', '60-69', '70+']
age_prob['dist'] = ['beta', 'beta', 'beta', 'beta', 'beta']


# In[15]:

age_prob['n'] = [n_0, n_1, n_2, n_3, n_4]


# In[16]:

age_prob


# In[17]:

age_prob = age_prob[['age', 'n', 'recurrence', 'dist']]


# In[18]:

p_recurrence = pd.DataFrame(data = age_prob)


# In[27]:

p_recurrence['recurrence'] = p_recurrence.recurrence.astype(float)


# In[28]:

p_recurrence


# In[29]:

p_recurrence.to_csv('M:\pc\Desktop\lit_review\p_recurrence.csv')


# In[ ]:




# # -fra 'at risk' til død

# Kjørte litteratursøk med følgende spesifikasjoner i Medline 130117:
# - ("mortality")OR"death)) AND "breast conserving treatment")OR"breast conserving surgery")OR"breast conserving therapy") AND "meta analysis"[PublicationType]) AND "review"[PublicationType]
# - fikk ut 33 Funn, desverre ingen som gav brukbare tall for overall survival/all-cause mortality (litt overraskende).
# 
# La inn "Norway" som et søkeparameter og fann denne studien:
# 
# *Hartman-Johnsen et al, "Survival is Better After Breast Conserving Therapy than Masectomy for Early Stage Breast Cancer: A Register-Based Follow up Study of Norwegian Women Primary Operated Between 1998 and 2008"*
# 
# Dette er kanskje beste kilde nå, men har muligens litt kort horisont:
# 
# #### Survival 5-year Overall Death:
# a = Age < 50y, $N_{t=0}$ = 1785, $X_{t=0,...,5}$ = 89
# 
# b = Age 50-69y, $N_{t=0}$ = 5539, $X_{t=0,...,5}$ = 232
# 
# c = Age >= 70, $N_{t=0}$ = 741, $X_{t=0,...,5}$ = 91
# 
# her er X et tilfelle av død, uansett årsak
# 
# Kan da bruke dette til å regne om til årlig sannsynlighet for død, uansett årsak. Dette er en veldig forenkling egentlig. CEHM skriver at sykdomsspesikk dødelighet kan estimeres fra kliniske studier og kombineres med ikke-sykdomsspesifikk dødelighet, med to tilnærminger:
#         - anta at sykdomsspesifikk dødelighet er additiv til dødelighet, uansett årsak,
#         - anta at sykdomsspesifikk dødelighet er en multiplikativ faktor.
# De skriver videre at forskeren må gjøre antagelser om hvordan sykdomsspesifikk dødelighet endres over tid. Dette er kanskje det viktigste akkurat her. Det antas ofte at gitt ingen forandring i en persons sykdomsstatus vil sykdomsspesifikk død iholde over resten av levetiden.
# 
# For kreftpasienter, derimot, er den vanlige antagelsen at sykdomsspesifikk dødelighet returnerer til den som en sykdomsfri person har ved denne alderen, etter eksempelvis 10 år.
# 
# ###### Beslutning: Bruker denne tilnærmingen nå, så kan man heller endre denne dersom folk mener det er en for streng antagelse. Kan klargjøre i paperet at man har gjort denne forenklingen - må også huske dette i sensitivitetsanalysen.
# 

# In[75]:

# transformering fra 5-års rate til 5 års sannsynlighet:
def rp(rate):
    rate = rate
    prob = (1-c.exp(-rate*1))
    return prob


# In[76]:

ar = rp(89/1785)
br = rp(232/5539)
cr = rp(91/741)


# In[77]:

# kjør disse gjennom prp funksjonen
pa_death = prp(ar,5)
pb_death = prp(br,5)
pc_death = prp(cr,5)


# In[78]:

p_recurrence


# In[79]:

p_recurrence['all-cause death'] = [pa_death, pa_death, pb_death, pb_death, pc_death]


# In[80]:

p_recurrence


# In[81]:

p_atrisk_noncarriers = p_recurrence


# In[20]:

p_atrisk_noncarriers.to_csv('M:\pc\Desktop\lit_review\p_atrisk_noncarriers.csv')


# # fra 'at risk' -> 'eggstokkreft'

# Denne må tolkes som sannsynligheten for at en person som har blitt gitt brystbevarende behandling utvikler eggstokkreft. Det er nok innafor å anta at denne tilsvarer sannsynligheten for at en helt frisk kvinne utvikler eggstokkreft. Dette fordi vi antar at brystkreften disse hadde var sporadisk, og da kan vi ikke etablere et kausalitetsforhold til andre kreftformer som har et slikt via genetisk arvelighet som bærere har. 
# 
# Benytter insidensrater (gjennsomsnitts) for eggstokkreft per 100000 i årene 2011-2015. https://www.kreftregisteret.no/globalassets/cancer-in-norway/2015/tabeller-statistikk/table-13.xlsx
# 
# 

# In[82]:

ovarian_rates = pd.DataFrame(columns= ['age', 'rate_per_100000'])


# In[83]:

ovarian_rates['age'] = ['40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85+']


# In[84]:

ovarian_rates['rate_per_100000'] = [8.1, 14.0, 29.8, 36.1, 46.1, 55.4, 55.7, 60.3, 54.4, 56.8]


# I modellen vil da den yngre probandkohorten først være underlagt risk for aldersgruppe 40-44, og etter som fem cycles er passert vil de gå over til neste aldersblokk, fem år etter det igjen er det risk i neste aldersblokk som gjelder osv.
# 
# #### Må nå transformere disse ratene per 100 000 til overgangssannsynligheter per år

# In[85]:

def ir(rate_per_100000):
    ir = rate_per_100000/100000
    return ir


# In[86]:

ir(ovarian_rates['rate_per_100000'])


# In[87]:

i = ir(ovarian_rates['rate_per_100000'])


# In[88]:

ovarian_rates['prob_of_oc'] = [rp(i[0]), rp(i[1]), rp(i[2]), rp(i[3]), rp(i[4]), rp(i[5]), rp(i[6]), rp(i[7]), rp(i[8]), rp(i[9])]


# In[89]:

ovarian_rates['prob_of_oc']= ovarian_rates.prob_of_oc.astype(float)


# In[90]:

ovarian_rates


# In[42]:

p_atrisk_to_oc = ovarian_rates
p_atrisk_to_oc['prob_of_oc'][0]


# In[43]:

p_atrisk_to_oc.to_csv('M:\pc\Desktop\lit_review\p_atrisk_to_oc.csv')


# # progression ovarian stage 1-4

# Eggstokkreft er et viktig endepunkt i modellen. Den bør behandles med eksakt samme detaljenivå som brystkreft. Dette kan bli en utfordring med tanke på data, fordi brystkreft sikkert er mer studert enn eggstokk på grunn av prevalensforskjellen. Uansett. Eggstokkreft er i de flese tilfeller epilelial eggstokkreft. Den er vanskelig å oppdage før det er blitt avansert kreft, 75% har spredning uten for det lokale området ved diagnosetidspunktet. På grunn av at det er dramatisk forskjell i forventet levetid hos de ulike stadiene av denne kreften kan det være hensiktsmessig å dele den opp i to tilstander;
# 
# a - stadium 1 og 2 - begrenset lokalt 
# 
# b - stadium 3 og 4 - regional og fjern spredning
# 
# Behandlingen i a vil som regel ha kurering som mål, mens behandlingen i b ofte er palliativ. 
# Disse behøver egentlig å ha fire muligheter hver; forbli i krefttilstand (ikke progrediering/progressjonsfri overlevelse), progressjon til b, eller død. b kan holde med progressjonsfri overlevelse, og død? Etter 10 år med progressjonsfri overlevelse kan man anta at de er statistisk kurert og kan overføres tilbake til 'at risk'. NB: her må man altså ha en semi-markov, fordi det vil spille stor rolle hvor lenge man har vært i tilstanden progressjonsfri... Desto lengre man holder ut her, desto høyere sannsynlighet for at man etterhvert ikke har noen nevneverdig høyere risiko for død enn normalbefolkningen.
# 
# Kanskje er dette for avansert per nå... Hva med tunneler?
# 
# Kjører på det først nå:
# 
# #### Dette kommer til å gjelde for ca. 25% av ovariekrefttilfellene (de som blir diagnostisert med stadium 1 og 2).
# 
# ##### Antagelse: 
# disse vil bli gitt såkalt makroskopisk radikal kirurgi (fjerning av alt synlig tumorvev) i sin første cycle i tilstanden, deretter vil de bli værende på et kjemoterapiregime så lenge de har progressjonsfri overlevelse https://legehandboka.no/handboken/kliniske-kapitler/gynekologi/tilstander-og-sykdommer/svulster-og-dysplasi/ovarialkreft/. Dersom de har gjennomgått 10 tunneller av progressjonsfri overlevelse antar vi at de er 'statistisk kurerte', hvorpå de overlevnde flytter tilbake til 'At risk', denne gang med kun risiko for brystkreft. For deres dødelighet vil vi også anta at denne går tilbake etter 10 år med progressjonsfri overlevelse, da 'relative survival' er oppe på 80% sammenlignet med frisk befolkning etter 10 år, se Cancer in Norway 2015, figur 100 side 83.
# 
# #### 75% av ovariekrefttilfellene vil ha avansert sykdom ved diagnosetidspunktet, (stadium 3 og 4). Her er det mindre sannsynlig at radikal kirurgi vil være suksessrik, så alternativet er å fjerne så mye som mulig, 'maksimal svulstreduksjon', og håpe på at cytostatika tar resten. 
# 
# ##### Antagelse:
# i og med at disse har en så pass avansert kreftform vil vi modellere de som om progressjon til et verre stadie ikke er mulig. Ideelt sett hadde vi kunnet stratifisert pasientene etter alle fire sykdomsstadier, men det finns veldig lite data som kan informere dette designet. Men vi antar også her, at dersom pasientene kommer overlever 10 år vil de være statistisk kurerte og kunne overføres tilbake i 'At risk', med påfølgende riskio for kun ny brystkreft.
# 
# 
# 
# ###### Husk senere at du må finne ut hvilke av tunnellene som har hvilke behandlingsfølger (kjemoterapi), viktig også er at ved failure i stadie 1-2 havner disse i den første tunnellen til stage 3-4 hvorpå en ny operasjon blir utført.

# ## Progression (recurrence/relapse)
# 
# Majoriteten av kvinner med avansert ovarialkreft (fra stadium II og opp) vil få tilbakefall og behøve ytterligere behandling. Ser man på alle stadiene under ett er livstidsrisiko for tilbakefall etter initiell behandling 62%. For de med de mest avanserte stadiene er riskioen 80-85%. Tilbakefall er en funksjon av stadium ved diagnose, hvorvidt primær kirurgisk behandling etterlater sykdom, og behandlingsresponsen etter primær behandling (plantinum sesitivitet/resistens). Vi har ingen perfekt perdiktor for tilbakefall ellers.
# 
# Kilde: https://www.uptodate.com/contents/medical-treatment-for-relapsed-epithelial-ovarian-fallopian-tubal-or-peritoneal-cancer-platinum-sensitive-disease?source=search_result&search=epithelial%20ovarian%20cancer%20recurrence&selectedTitle=1~150.
# 
# - For pasienter med lite residualsykdom < 1 cm i størrelse etter kirurgi er risiko (les:sannsynlighet) for tilbakefall etter fullført primærbehandling 60-70%. 
# - For pasitenter med mye residualsykdom, >= 1 cm, er risikoen estimert til 80-85%.
# 
# - Seleksjon av terapi for kvinner med tilbakefall er til stor del avhengig av deres respons på førstelinjebehandlingen. 
#     - Tilbakefalt ovarialkreft har blitt todelt til enten 'platinumsensitiv' (progressjonsfri > 6 måneder) eller 'platinumresisten' (progressjonsfri <= 6 måneder).
#     - Progressjonsfri tid predikerer forventet respons. Uavhengig av valgt behandling er tilbakefallende ovarialkreft fortsatt ikke mulig å kurere.
# 
# Kilde: https://www.ncbi.nlm.nih.gov/pubmed/23781692
#     
# ### Trenger solide tall på:
# - 'andel' med low/high residual disease/tumor (helst fra Norge i nyere tid)
# - risiko for tilbakefall (risk of recurrece/relapse)for begge kategoriene
# - hvorvidt de med tilbakefall er plantinumsensitive/resistente.
# 
# ##### Antar at de som ikke har fått tilbakefall etter minst 6 måneder er platinum sensitive, og motsatt de som har hatt tilbakefall er resistente.

# ## Risk of ovarian cancer relapse score (ROVAR)
# Man kan predikere OS ganske bra nå, men man har vært ganske på bakfoten hva gjelder det å predikere risiko for tilbakefall. En gruppe tilknyttet NHS(?) utviklet en indeks for å predikere denne riskioen via variabler de fant å være svært signifikante for tilbakefall i en logistisk regressjon av et stort datasett. De delte dette i to, og hadde et for å validere sin indeks, og fann bra prediksjonsverdi. 
# 
# Jeg vil benytte informasjon om norske EOC pasienter fra Kreftregisterets rapport 'Nasjonalt kvalitetsregister for gynekologisk kreft' til å regne meg fram til en trolig 5-års risiko for å få tilbakefall etter primærbehandling, og dermed kunne finne trolige verdier for en gjennomsnittspasient til hvert FIGO-stadie i modellen.
# 
# 

# In[70]:

# ROVAR INDEX:
r_noresid = pd.read_excel('M:/pc/Desktop/lit_review/ovarian/transitions/recurrence/rovar_to_py_noresid.xlsx')
r_anyresid = pd.read_excel('M:/pc/Desktop/lit_review/ovarian/transitions/recurrence/rovar_to_py_anyresid.xlsx')


# In[71]:

r_noresid # for patients with no macroscopic residual tumor


# In[72]:

r_anyresid # for patients with any macroscopic residual tumor


# #### Fra Norsk Kvalitetsregister for Gynekologisk Kreft - Årsrapport 2015:

# In[73]:

# Norske pas med ovarialkreft rapportert til Kreftregisterets kvalitetsregister - dekningsgrad 68 %

FIGO_1 = 88 # antall innrapportert tdiagnostisert med stadie 1 i 2015 i Norge
FIGO_2 = 11 # antall innrapportert diagnsotisert med stadie 2 i 2015 i Norge
FIGO_3 = 158 # antall innrapportert diagnostisert med stadie 3 i 2015 i Norge
FIGO_4 = 65 # antall innrappoerter diagnostisert med stadie 4 i 2015 i Norge
N = FIGO_1 + FIGO_2 + FIGO_3 + FIGO_4 # antall inrapportert til Kreftregisteret
f1 = FIGO_1/N
f2 = FIGO_2/N
f3 = FIGO_3/N
f4 = FIGO_4/N


# In[259]:

p_ocstage_at_diagn0 = pd.DataFrame(data = [(f1, f2, f3, f4)], columns = ('stage 1', 'stage 2', 'stage 3', 'stage 4'))
p_ocstage_at_diagn0.to_csv('M:\pc\Desktop\lit_review\p_ocdiag_to_stage.csv')


# In[75]:

# Norske pas med residualtumor etter primærbehandling fra samme reg (kun for FIGO 2-4 - dekningsgrad 45 %)

No_resid = 69 # antall med ingen restttumor
Any_resid = 63 # antall med minst < 1 cm resttumor
N2 = No_resid + Any_resid
r0 = No_resid/N2
r1 = 1 - r0


# In[76]:

# Norske pas med CA-125 > 35 ved klinisk utredning 2015(Over normalverdier, pred for ondartethet - dekningsgrad 60 %)
ca_125_lo = 37
ca_125_hi = 272
N3 = ca_125_lo + ca_125_hi
c0 = ca_125_lo/N3
c1 = 1 - c0


# In[77]:

# Malignitetsgrad (1-3) ikke oppgitt i kvalitetsregisteret
# i Artiklen 'Prognostic Value of Pre and Postoperative Serum CA125 Levels...' Makar et al, hadde 481 pasienter behandlet
# ved Radiumhospitalet følgende malignitetsgrader:
G1 = 57 # Antall med grad 1
G2 = 146 # Antall med grad 2
G3 = 278 # Antall med grad 3
N4 = G1 + G2 + G3
g1 = G1/N4
g2 = G2/N4
g3 = G3/N4

# Denne er litt tricky...


# OK! Hvis vi nå skal prøve å komme fram til én sannsynlighet etter FIGO stadium må vi altså vekte etter:
# - Sannsynlighet for å ha residualtumor eller ikke, 2 muligheter
# - Sannsynlighet for å ha forhøyd CA125, 2 muligheter
# - Sannsynlighet for å ha de ulike gradene av tumormalignitet, 3 muligheter
# 
# Hvert FIGO-stadium har altså 12 mulige sannsynligheter!
# Vil dermed benytte tallene fra Kreftregisteret til å vekte ROVAR utfallene, og finne et vektet gjennomsnitt (med standardavvik) for hvert FIGO stadie.

# ### For FIGO 1

# In[80]:

figo1_nr = [r_noresid.ca_125_low[0], r_noresid.ca_125_low[1], r_noresid.ca_125_low[2], r_noresid.ca_125_high[0], r_noresid.ca_125_high[1], r_noresid.ca_125_high[2]]


# In[81]:

wf1_nr = [(c0*g1), (c0*g2), (c0*g3), (c1*g1), (c1*g2), (c1*g3)]


# In[82]:

f1_nr = np.multiply(figo1_nr, wf1_nr)


# In[83]:

f1_nr # All possible weighted probabilites for FIGO 1 patients who have no residual tumor


# In[84]:

figo1_r =[r_anyresid.ca_125_low[0], r_anyresid.ca_125_low[1], r_anyresid.ca_125_low[2], r_anyresid.ca_125_high[0], r_anyresid.ca_125_high[1], r_anyresid.ca_125_high[2]]


# In[85]:

f1_r = np.multiply(figo1_r, wf1_nr)


# In[86]:

f1_r # All possible weighted probabilities for FIGO 1 patients who have a residual tumor of any visible size


# In[87]:

figo1_nr_mu = np.sum(f1_nr)/np.sum(wf1_nr)
figo1_nr_sigma = np.std(figo1_nr, ddof=1)
figo1_nr_mu


# In[88]:

figo1_r_mu = np.sum(f1_r/np.sum(wf1_nr))
figo1_r_sigma = np.std(figo1_r, ddof=1)
figo1_r_mu


# Now we can further find the 5-year probability that a random Norwegian FIGO 1 patient would have a relapse by weighting the probabilties for either being in the 'no residual disease' category or the 'any residual disease' category by the probability of being in those categories. We know that its ~50/50 likely that there is *some* residual disease, and that the probability of relapse is high for this group, so the an average FIGO 1 patient would probably be a fair bit more likely to relapse than those with no residual disease. 

# In[89]:

figo1_mu0 = (figo1_nr_mu*r0) + (figo1_r_mu*r1)/(r0+r1)


# In[90]:

figo1_mu0


# In[91]:

figo1_sigma0 = (figo1_nr_sigma*r0) + (figo1_r_sigma*r1)/(r0+r1)
figo1_sigma0


# Transform from 5 year probabiltiy to 1 year probabiltiy (through function prp above)

# In[92]:

figo1_mu = prp(figo1_mu0,5)
figo1_sigma = prp(figo1_sigma0,5)


# ### For FIGO 2

# In[93]:

figo2_nr = [r_noresid.ca_125_low[4], r_noresid.ca_125_low[5], r_noresid.ca_125_low[6], r_noresid.ca_125_high[4], r_noresid.ca_125_high[5], r_noresid.ca_125_high[6]]


# In[94]:

f2_nr = np.multiply(figo2_nr, wf1_nr)


# In[95]:

figo2_nr_mu = np.sum(f2_nr)/np.sum(wf1_nr)
figo2_nr_sigma = np.std(figo2_nr, ddof=1)
figo2_nr_mu


# In[96]:

figo2_r =[r_anyresid.ca_125_low[4], r_anyresid.ca_125_low[5], r_anyresid.ca_125_low[6], r_anyresid.ca_125_high[4], r_anyresid.ca_125_high[5], r_anyresid.ca_125_high[6]]


# In[97]:

f2_r = np.multiply(figo2_r, wf1_nr)


# In[104]:

figo2_r_mu = np.sum(f2_r)/np.sum(wf1_nr)
figo2_r_sigma = np.std(figo2_r, ddof=1)
figo2_mu0 = (figo2_nr_mu*r0) + (figo2_r_mu*r1)/(r0+r1)
figo2_mu0
figo2_r_mu


# In[106]:

figo2_mu0 = ((figo2_nr_mu*r0) + (figo2_r_mu*r1))/(r0+r1)
figo2_mu0


# In[108]:

figo2_sigma0 = ((figo2_nr_sigma*r0) + (figo2_r_sigma*r1))/(r0+r1)
figo2_sigma0


# In[109]:

figo2_mu = prp(figo2_mu0,5)
figo2_sigma = prp(figo2_sigma0,5)


# ### For FIGO 3

# In[110]:

figo3_nr = [r_noresid.ca_125_low[8], r_noresid.ca_125_low[9], r_noresid.ca_125_low[10], r_noresid.ca_125_high[8], r_noresid.ca_125_high[9], r_noresid.ca_125_high[10]]


# In[111]:

f3_nr = np.multiply(figo3_nr, wf1_nr)


# In[112]:

figo3_nr_mu = np.sum(f3_nr)/np.sum(wf1_nr)
figo3_nr_sigma = np.std(figo3_nr, ddof=1)
figo3_nr_mu


# In[113]:

figo3_r =[r_anyresid.ca_125_low[8], r_anyresid.ca_125_low[9], r_anyresid.ca_125_low[10], r_anyresid.ca_125_high[8], r_anyresid.ca_125_high[9], r_anyresid.ca_125_high[10]]


# In[114]:

f3_r = np.multiply(figo3_r, wf1_nr)


# In[115]:

figo3_r_mu = np.sum(f3_r)/np.sum(wf1_nr)
figo3_r_sigma = np.std(figo3_r, ddof=1)
figo3_r_mu


# In[116]:

figo3_mu0 = (figo3_nr_mu*r0) + (figo3_r_mu*r1)/(r0+r1)
figo3_mu0


# In[117]:

figo3_sigma0 = (figo3_nr_sigma*r0) + (figo3_r_sigma*r1)/(r0+r1)
figo3_sigma0


# In[118]:

figo3_mu = prp(figo3_mu0,5)
figo3_sigma = prp(figo3_sigma0,5)


# ### For FIGO 4

# In[119]:

figo4_nr = [r_noresid.ca_125_low[12], r_noresid.ca_125_low[13], r_noresid.ca_125_low[14], r_noresid.ca_125_high[12], r_noresid.ca_125_high[13], r_noresid.ca_125_high[14]]


# In[120]:

f4_nr = np.multiply(figo4_nr, wf1_nr)


# In[121]:

figo4_nr_mu = np.sum(f4_nr)/np.sum(wf1_nr)
figo4_nr_sigma = np.std(figo4_nr, ddof=1)
figo4_nr_mu


# In[122]:

figo4_r =[r_anyresid.ca_125_low[12], r_anyresid.ca_125_low[13], r_anyresid.ca_125_low[14], r_anyresid.ca_125_high[12], r_anyresid.ca_125_high[13], r_anyresid.ca_125_high[14]]


# In[123]:

f4_r = np.multiply(figo4_r, wf1_nr)


# In[124]:

figo4_r_mu = np.sum(f4_r)/np.sum(wf1_nr)
figo4_r_sigma = np.std(figo4_r, ddof=1)
figo4_r_mu


# In[125]:

figo4_mu0 = (figo4_nr_mu*r0) + (figo4_r_mu*r1)/(r0+r1)
figo4_mu0


# In[126]:

figo4_sigma0 = (figo4_nr_sigma*r0) + (figo4_r_sigma*r1)/(r0+r1)
figo4_sigma0


# In[127]:

figo4_mu = prp(figo4_mu0,5)
figo4_sigma = prp(figo4_sigma0,5)


# ### Export

# In[128]:

p_recurrence_by_ocstage = pd.DataFrame(data=[(figo1_mu, figo1_sigma), (figo2_mu, figo2_sigma), (figo3_mu, figo3_sigma), (figo4_mu, figo4_sigma)], index = ('FIGO stage 1', 'FIGO stage 2', 'FIGO stage 3', 'FIGO stage 4'), columns= ('1-year probability (mu)', 'std'))


# In[129]:

p_recurrence_by_ocstage


# In[130]:

p_recurrence_by_ocstage.to_csv('M:\pc\Desktop\lit_review\p_recurrence_by_ocstage.csv')


# # to death from OC

# Maringe et al *Stage at diagnosis and ovarian canser survival: Evidence from the International Cancer Benchmarking Partnership* kan man finne 1-års OS rater for norske OC-pasienter etter alder og FIGO stadium. Tallene er hentet ut fra Kreftregisteret av en forsker derfra og dekker årene 04-07. 

# In[131]:

# Survival rates Norway OC, by age and FIGO stage
OC_OS = pd.read_excel('M:/pc/Desktop/lit_review/ovarian/transitions/to death/OCOS.xlsx')


# In[132]:

OC_OS


# In[133]:

OC_OS = OC_OS.drop(['OS-rate', '95% CI low', '95%CI high', 'Haz-rate', '95% CI low.1', '95%CI high.1'], axis=1)


# In[134]:

p_from_OCstage_to_death = OC_OS.to_csv('M:/pc/Desktop/lit_review/p_from_OCstage_to_death.csv')


# In[ ]:




# # at risk -> breast cancer (relatives) 

# Uten genetisk informasjon må vi anta at denne utvikles som hos de sporadiske tilfellene i den generelle norske befolkningen. Går derifor utifra at Kreftregisterets tall fra Insidensregisteret må kunne tolkes som den beste tilgjengelige kilde. 
# 
# For at 'relatives' er både datter og søster tar jeg ut alle alderskategoriene hvor der er registrert insidenser med variablen *'age-specific incidence rates per 100000 person-years by primary site and five-year age-gropu, 2011-2015*.
# 
# Denne vil kunne transformeres først fra 5års rate per 100000 til 1års insantaneous rate, og derifra til årlig sannsynlighet.
# 
# Kilde:
# Insidensrater (gjennsomsnitts) for brystkreft per 100000 i årene 2011-2015. https://www.kreftregisteret.no/globalassets/cancer-in-norway/2015/tabeller-statistikk/table-13.xlsx

# In[44]:

breastc_rates = pd.DataFrame(columns= ['age', 'rate_per_100000'])


# In[45]:

breastc_rates['age'] = ['15-19','20-24', '25-29', '30-34', '35-39','40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85+']


# In[46]:

breastc_rates['rate_per_100000'] = [0.1, 1.9, 10.9, 23.2, 59.1, 109.0, 171.4, 238.3, 248.1, 317.2, 348.4, 257.3, 292.8, 286.5, 286.0]


# In[47]:

breastc_rates


# In[48]:

i2 = ir(breastc_rates['rate_per_100000'])


# In[49]:

breastc_rates['rate'] = i2


# In[50]:

breastc_rates['prob_of_bc'] = [rp(i2[0]), rp(i2[1]), rp(i2[2]), rp(i2[3]), rp(i2[4]), rp(i2[5]), rp(i2[6]), rp(i2[7]), rp(i2[8]), rp(i2[9]), rp(i2[10]), rp(i2[11]), rp(i2[12]), rp(i2[13]), rp(i2[14])]


# In[51]:

breastc_rates['prob_of_bc'] = breastc_rates.prob_of_bc.astype(float)


# In[ ]:




# In[52]:

p_atrisk_to_bc = breastc_rates


# In[53]:

p_atrisk_to_bc


# In[54]:

p_atrisk_to_bc.to_csv('M:\pc\Desktop\lit_review\p_atrisk_to_bc.csv')


# In[55]:

pd.read_csv('M:\pc\Desktop\lit_review\p_atrisk_to_bc.csv')


# # at-risk -> all cause / background mortality (relatives)

# Use the approach in Hunik p.338 in which a *Declining exponential approximation of life expectancy* (DEALE) is used. Here we assume a constant hazard function h(t) = $\mu$.
# 
# $\mu_{Total}$ is decomposed into a general population mortality rate, $\mu_{ASR}$ given age, sex, and race, and a disease-specific mortality rate, $\mu_{Disease}$.
# 
# For this all cause mortality, we set it equal to the population mortality rate since we assume no illness in the at-risk state. 
# 
# To calculate the $\mu_{ASR}$ we can utilize the life-expectancy given age from official life-tables:
# $\mu_{ASR}$ = 1/LE(age, sex, race). 
# 
# Following the calculation of the general population mortality rate we can calculate the probability of dying given age required for a cycle length of dt in the state transition model as 
# 
# $p_{Die}$ = (1-exp(-$\mu_{ASR}$ * dt)
# 
# From Statistics Norway we have the life expectancy for women in 2015 in table 07902,
# http://www.ssb.no/en/table/07902.

# In[183]:

life_exp = pd.read_excel('M:/pc/Desktop/lit_review/breast/atrisk_death/life_exp_2015.xlsx')


# In[184]:

life_exp


# In[54]:

life_exp.to_csv('M:\pc\Desktop\lit_review\p_die_bgpop.csv')


# # Breast cancer stage at diagnosis (relatives)
# PubMed lit søk i mappe. 1 bra studie funnet. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3619080/
# 
# Tall fra Norge, Kreftregisteret årene 2000 - 2007.
# 
# N = 21 964 kvinner. 2873 hadde 'stage missing'. Forfatterne imputerte trolige verdier gjennom 'imputation by chained equations' (ICE). Etter fordeling er kummulativ fordeling etter alder og stadium: 

# In[96]:

p_stage_age = pd.DataFrame(columns = ['age', 'stageI', 'stageII', 'stageIII', 'stageIV'])


# In[97]:

p_stage_age['age'] = ['15-49', '50-69', '70-99' ]


# In[98]:

p_stage_age['stageI'] = [0.345, 0.571, 0.339]
p_stage_age['stageII'] = [0.576, 0.363, 0.492]
p_stage_age['stageIII'] = [0.035, 0.021, 0.065]
p_stage_age['stageIV'] = [0.044, 0.045, 0.104]
p_stage_age


# In[99]:

p_stage_age.to_csv('M:\pc\Desktop\lit_review\p_dist_age_stage_bc_relavives.csv')


# In[ ]:




# ### *Update*
# 
# På grunn av mangel på data vil det ikke være mulig *nå* å modellere hvert stadium med spesifikke overgangssannsynligheter til død og progressjon. Må gjøre en pragmatisk forenkling og holde brystkreft på 'early stage' og 'distant/late stage'...
# 
# Kwon et al gjorde følgende: "(...) women diagnosed with breast cancer transition to the breast cancer state, which is determined by their age, BRCA mutation status, and whether they had profylactic surgery. In this state they may
# - die of breast cancer,
# - or die of age-dependent competing mortalities.
# If they are alive after a year they remain in that state and are subject to cancer-specific and age-dependent mortalities."
# 
# "(...) if diagnosed with ovarian cancer they transition to the ovarian cancer state where they are s.t. cancer-specific mortality (30 % lower in BRCA mutation carriers)."
# 
# 
# Jeg vil gjøre følgende:
# - Breast cancer states vil være 'én for tidlig, én for avansert (*hvis jeg kan finne noe...*), hvorpå det ikke blir mulig å flyte mellom stadiene, man er kun s.t. dødelighet. De vil forbli i disse helt til de har vært igjennom 10 cycles, hvorpå tablettbehandling avsluttes, og dersom de fortsatt lever går tilbake til 'at risk'.
# - Dødeligheten vil være total, dvs. bc-specific + background.
# 

# Kilde for tall: 
# https://www.ncbi.nlm.nih.gov/pubmed/25743325

# In[193]:

# 5 -year breast cancer mortality = 1 - 5-year breast cancer survival
# Patients treated with breast conserving therapy, aged < 50 years
bct_age_lt50 = 0.04

# Patients treated with breast conserving therapy, aged 50 - 69 years
bct_age_5069 = 0.02

# Patients treated with breast conserving therapy, age > 70 years
bct_age_mt70 = 0.06

# Patients treated with masectomy, aged < 50 years
mtx_age_lt50 = 0.12

# Patients treated with masectomy, age 50 - 69 years
mtx_age_5069 = 0.1

# Patients treated with masectomy, age > 70 years
mtx_age_mt70 = 0.14


# In[194]:

# Function to change length of time, rate (t = t year)-> prob -> rate (t = 1 year)
def rpr(rate, t):
    rate_t = rate
    prob = (1-c.exp(-rate*1))
    rate_1 = -c.log(1-prob)/t
    return rate_1


# In[197]:

#excess_bct_age_lt50 = rpr(bcm_age_lt50, 5)
#excess_bct_age_5069 = rpr(bcm_age_5069, 5)
#excess_bct_age_mt70 = rpr(bcm_age_mt70, 5)

#excess_mtx_age_lt50 = rpr(mtx_age_lt50, 5)
#excess_mtx_age_5069 = rpr(mtx_age_5069, 5)
#excess_mtx_age_mt70 = rpr(mtx_age_mt70, 5)

#print(excess_bct_age_lt50, excess_bct_age_5069, excess_bct_age_mt70, excess_mtx_age_lt50, excess_mtx_age_5069, excess_mtx_age_mt70 )


# In[199]:

# Fiksa resten i Excel... tok for lang tid å mekke i Py
p_death_es = pd.read_excel('M:/pc/Desktop/lit_review/breast/mortality.xlsx')


# In[202]:

p_death_es = p_death_es[20:] # NB! Dette er kun 'early stage (I & II)


# In[203]:

p_death_es


# ## *Approach over faller sammen fordi jeg ikke kan finne noen fornuftige estimat på total overlevelse i litteraturen for lokalavansert og metastasert brystkreft. Ny approach:
# 
# #### - tar ut relativ 5-års overlevelse fra Kreftregisterets Cancer in Norway 2015.
# #### - tilpasser til rate, beregner derfor alders - og stadiumbetingede dødelighetsrater ved å kombinere relativ (merk! ikke excess/overskuddsdødelighet av sykdom, kun relativ til en alders, kjønn og rase matchet generell populasjon) og bakgrunnsdødelighet. 

# In[56]:

rs_bc = pd.DataFrame(columns = ['Stage','RS (%)'])
rs_bc['Stage'] = ['I', 'II', 'III', 'IV']
rs_bc['RS (%)'] = [100.2, 92.3, 76.0, 25.5]
rs_bc


# Ser at relativ 5-års overlevelse for pasienter med brystkreft stadium 1 er > 100 %, det vil bedre enn hos referanse populasjone. Dette er selvsagt kontraintuitivt da det er lite trolig at ens overlevelse øker ved å få brystkreft stadium 1, alt annet like. Brenner og Arndt spekulerte i sin '*Long-term survival rates of patients with prostate cancer in the prostate-specific antigen era: population-based estimates for the year 2000 by period analysis*' at en mer plausibel forklaring er seleksjonseffekter på grunn av screening. Det vil si, de som velger å delta i screening har større sannsynlighet for å være av høyere sosioøkonomisk status enn de som velger å stå utenfor, slik at det er ikke sykdommen som forklarer den høyere overlevelsen, det er faktumet at disse pasientene er i utgangspunktet friskere, eller har generelt lavere dødelighet. 
# 
# Jeg tror det er nettopp dette fenomenet som skylder RS > 100 % for denne gruppen, da majoriteten sannsynligvis ble diagnostisert som følge av deltagelse i mammografiprogrammet. 
# 
# For å løse dette velger jeg å inkludere RS'en fra den tidligere 5-årsperioden, 06-10, og ta et gjennomsnitt av disse.
# -> *(99.6+100.2)/2 = 99.9%*.

# #### - følger nå Hunik's approach (p.340);
# $\mu_{Total}$ = $\mu_{ASR}$ + $\mu_{D}$
# 
# hvor
# 
# $\mu_{D}$ (sykdomsspesifikk dødelighet) = -ln(RS(t))/t
# 
# #### - finner dermed en approksimering til overgangssannsynligheter til død fra brystkreft over aksene stadium og alder.

# In[57]:

rs_bc['RS (%)'] = [99.9, 92.3, 76.0, 25.5]
rs_bc['RS'] = [99.9/100, 92.3/100, 76.0/100, 25.5/100]


# In[58]:

rs_bc


# In[59]:

def rs_r(rs, t):
    r = -c.log(rs)/t
    return r


# In[60]:

rs_bc['excess_r_stage'] = [rs_r(rs_bc['RS'][0], 5), rs_r(rs_bc['RS'][1], 5), rs_r(rs_bc['RS'][2], 5), rs_r(rs_bc['RS'][3], 5)]


# In[62]:

rs_bc['excess_r_stage'] = rs_bc.excess_r_stage.astype(float)


# In[63]:

rs_bc


# In[255]:

# igjen, mekket ihop resten i Excel
p_die_bc = pd.read_excel('M:/pc/Desktop/lit_review/breast/mortality0.xlsx')


# In[256]:

p_die_bc


# In[257]:

p_die_bc.to_csv('M:\pc\Desktop\lit_review\p_die_bc.csv')


# In[258]:

# ps. når/hvis du skal importere disse i excel senere gå til 'data', 
# så til 'fra tekst', velg 'komma' (CSV = kommaseparerte verdier).


# # Samme pros. for OC

# In[101]:

# Kilde cancer in Norway (nei, se under)


# In[103]:

p_ocstage_at_diagn = pd.DataFrame(columns = ['stage', 'dist'])


# In[104]:

p_ocstage_at_diagn['stage'] = ['localised', 'regional', 'distant']
p_ocstage_at_diagn['%'] = [0.218, 0.031, 0.683]

# !!NB, kan ikke bruke disse da vi ikke har unity, pga at en andel pas. ikke har fått sine stadier.
# Må desverre derfor bruke eldre verdier (2004 - 2007) her er missing imputert, tall fra KR Norge.
# https://www.ncbi.nlm.nih.gov/pubmed/22750127
    
p_ocstage_at_diagn['dist'] = [0.121, 0.182, 0.698]
p_ocstage_at_diagn


# In[105]:

p_ocstage_at_diagn.to_csv('M:\pc\Desktop\lit_review\p_ocdiag_to_stage.csv')


# #### Relative 5-year survival

# In[64]:

rs_oc = pd.DataFrame(columns = ['Stage','RS'])
rs_oc['Stage'] = ['localized', 'regional', 'distant']
rs_oc['RS'] = [0.932, 0.616, 0.342]


# In[65]:

rs_oc['excess_r_stage'] = [rs_r(rs_oc['RS'][0], 5), rs_r(rs_oc['RS'][1], 5), rs_r(rs_oc['RS'][2], 5)]


# In[67]:

rs_oc['excess_r_stage'] = rs_oc.excess_r_stage.astype(float)
rs_oc


# In[320]:

# igjen, mekket ihop resten i Excel
p_die_oc = pd.read_excel('M:/pc/Desktop/lit_review/ovarian/mortality1.xlsx')


# In[321]:

p_die_oc


# In[322]:

p_die_oc.to_csv('M:\pc\Desktop\lit_review\p_die_oc.csv')


# In[ ]:



