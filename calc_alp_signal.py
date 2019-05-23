import numpy as np
from scipy.interpolate import UnivariateSpline as USpline
from scipy.interpolate import RectBivariateSpline as RBSpline
from scipy.integrate import simps
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import argparse
import logging
from gammaALPs import Source, ALP, ModuleList
from astropy.table import Table, vstack
from collections import OrderedDict

class ALPSNSignal(object):
    M18par = np.array([np.concatenate(([0.005,0.2,0.5,1.,1.5],
                np.arange(2.,19.,1.))),
                [6.24e-2,3.94e-1,8.05e-1, 1.03,1.1,1.11,1.07,9.85e-1,
                 8.82e-1,7.74e-1,6.66e-1,5.68e-1,4.85e-1,4.11e-1,3.49e-1,2.98e-1,
                 2.53e-1,2.14e-1,1.78e-1,1.55e-1,1.3e-1,1.11e-1],
                [35.2,77.3,98.5,105.6,107.6,108.3,107.8,106.7,
                 105.3,103.9,102.4, 100.8, 99.4,97.5, 95.8,93.7,91.6,89.5, 
                 87.2,85.8, 82.9, 80.2],
                [2.25,2.02,2.065, 2.145,2.19, 2.22,2.28, 2.315, 2.34, 2.35, 2.355,
                2.355, 2.35, 2.35, 2.35, 2.35, 2.35, 2.35, 2.355, 2.355, 2.37, 2.385]
                 ]).T
# string in
#t[s]        C[10^52/MeV/s]        E0[MeV]        beta
    M10par = np.array([np.fromstring(f,dtype = np.float, count = 4, sep = '\t') \
    for f in """0.1        0.18356234297932772        59.225788057998315        2.111893326954856
0.2        0.33265859234613276        74.47928172588773        2.124303164836993
0.30000000000000004        0.495810509478926        85.24974115719753        2.1483700306909013
0.4        0.6267544529772382        91.67828238555907        2.1787015325763837
0.5        0.6945518913829549        94.45143514496476        2.2039149492831127
0.6        0.753753702243041        96.48856637335031        2.2321157960740643
0.7000000000000001        0.7963192208975821        98.03525142039486        2.254851190063449
0.8        0.8271972758357318        99.20403426809571        2.2728136237103604
0.9        0.844403351193037        99.88487323566399        2.2851935003715877
1.        0.8619978990653584        100.58262687758643        2.3007575084564356
1.1        0.8746235443300421        101.06529197747183        2.3132458188800062
1.2000000000000002        0.8830112328839931        101.43477453534477        2.323141760652145
1.3000000000000003        0.8880090962234579        101.71617511740419        2.336568434062643
1.4000000000000001        0.8907765521705368        101.93578983558997        2.346121619954014
1.5000000000000002        0.8934546637624584        102.1275518841011        2.3543888580799157
1.6        0.8916893833221207        102.23553665405166        2.3656956702268346
1.7000000000000002        0.8885835633757634        102.2661843917327        2.3726794354549776
1.8000000000000003        0.8856735733133517        102.3482308449654        2.3828091179533524
1.9000000000000001        0.8822684681537318        102.36169196528778        2.3889157834858223
2.        0.8735755933572245        102.24009684713448        2.397582536249191
2.1        0.8682424446759958        102.02985969658909        2.4043247796269314
2.2        0.8562098543834873        101.87417333202416        2.4138141682438126
2.3000000000000003        0.8495032409172905        101.79595497568035        2.419142086714644
2.4000000000000004        0.8422766271853632        101.68770972226868        2.4247499490291857
2.5000000000000004        0.8345683733061584        101.60353695662099        2.4306098318039218
2.6        0.8233715939142091        101.45545395005664        2.436524177235097
2.7        0.8134128753265025        101.20724234545291        2.4428107736262015
2.8000000000000003        0.804154731809432        101.084422208953        2.449500224424181
2.9000000000000004        0.7970935401771944        100.98185387797301        2.453159574340271
3.0000000000000004        0.7873089949026961        100.83502208459846        2.4585268943510235
3.1        0.7774904473529332        100.6741994455164        2.463568602944285
3.2        0.7660816007611422        100.49200166308243        2.469247455183771
3.3000000000000003        0.7551842853157374        100.32172454813568        2.472504013782174
3.4000000000000004        0.7463395782526153        100.17391276502904        2.4766582508659396
3.5000000000000004        0.7369377697591861        100.02765556823007        2.480558033093819
3.6        0.7263217916272825        99.83024068911642        2.4858019031776886
3.7        0.7152792173101087        99.67750902930847        2.489145129490189
3.8000000000000003        0.7045033031026494        99.51889930041493        2.492850099530138
3.9000000000000004        0.6943573324357967        99.34227609152289        2.496611491287445
4.        0.6829156930098712        99.04442987396499        2.500637687055382
4.1        0.67368519235856        98.88558434539412        2.5034166175775865
4.2        0.6627259791225388        98.7042018961846        2.5065402479884114
4.3        0.6513111413770475        98.51660640564977        2.508497135990119
4.3999999999999995        0.6414452423789342        98.36835963576934        2.510646938317136
4.5        0.631106576056166        98.18903852114498        2.513685646304643
4.6        0.6248498905435532        98.08336617761181        2.515250367369756
4.7        0.6096842461489727        97.82437997550373        2.5179027831667393
4.8        0.6036704385119731        97.74409821293318        2.519556543650331
4.9        0.5945740124310682        97.60065894143811        2.521331305872974
5.        0.5842864855130566        97.44283677744744        2.5237584815365635
5.1        0.5716030356626627        97.24725916679293        2.5248747902428614
5.2        0.5640326558141637        97.10824857123049        2.5266683246580732
5.3        0.5517078891107557        96.74754220205492        2.5297581786045082
5.4        0.5408803525691386        96.57027174942652        2.5310566633350877
5.5        0.5328797244225091        96.43471803797415        2.5320259637458826
5.6        0.5244904277089681        96.31904749809688        2.533061398512544
5.7        0.5143633800825086        96.14251437054745        2.534089983269482
5.8        0.5063508050210245        96.01185862064727        2.534818832317848
5.9        0.497781715418321        95.86850549509731        2.53588793048487
6.        0.48953288997799954        95.73721815839325        2.536508238091037
6.1        0.481498501182188        95.61034504336745        2.5370360274584742
6.2        0.4735506397539696        95.48003898627285        2.5374172516706692
6.3        0.4654791691861431        95.34405412137814        2.537919716409935
6.4        0.45801712682277923        95.2451530611409        2.5385143407294963
6.5        0.4498171516007783        95.14335334215214        2.538596768589397
6.6        0.441541808415432        95.00982057491694        2.5388193091636637
6.7        0.43389579698380665        94.88870685396701        2.538955492247458
6.8        0.42615943465969        94.77087036519704        2.5388401149161024
6.9        0.4187609160763958        94.65226652805549        2.5387310214144034
7.        0.41145352110494315        94.52561774243205        2.5385981104316833
7.1        0.40353655419536877        94.41878538334247        2.5388886466142444
7.2        0.3964927711463429        94.28995138352282        2.5388749938859987
7.3        0.3892617096656783        94.18015083028658        2.5385662716911894
7.4        0.3822743857047242        94.05053003613077        2.5384067380383453
7.5        0.3752016033809963        93.92415668393873        2.537924511240994
7.6        0.36824654857756417        93.79137141509325        2.537492373469741
7.7        0.36264697662988565        93.69093192308013        2.5370135175891573
7.8        0.3561034886212772        93.57471688394521        2.5366728295714296
7.9        0.3510507093499212        93.63445622439609        2.5355572469759244
8.        0.34484057450383304        93.52542872463903        2.5350283892013787
8.1        0.3374862593441133        93.3447881447894        2.534606677153899
8.2        0.33186711246116896        93.25914860380864        2.53458041517633
8.3        0.3260115772076773        93.14624590989295        2.5340099795409654
8.4        0.3212114032446858        93.03387968428758        2.533953965191684
8.5        0.31598368939981397        92.92868441980477        2.533541242296977
8.6        0.3105276069616932        92.8245177410627        2.5337609358211552
8.7        0.30260263400311177        92.65315438388586        2.533022429962727
8.8        0.297332894011627        92.53597758956545        2.532298127401243
8.9        0.29194397229947516        92.40758955407425        2.5315208635847344
9.        0.28456272670473975        92.30610956830736        2.5303608522858885
9.1        0.27798089653952957        92.16168102992208        2.5290602460993012
9.2        0.27267358365733135        92.03600170944789        2.528266552368913
9.3        0.2677031552722398        91.91546758863008        2.527428392413964
9.4        0.26320196142497554        91.80089266744102        2.526757491530212
9.5        0.2584114996115843        91.67003174111913        2.5262730348299116
9.6        0.25375001130984237        91.56470233642415        2.526179505100624
9.700000000000001        0.2489653137294185        91.45060408699936        2.5252497031241403
9.8        0.24404088591910433        91.32916894038405        2.524134943180253
9.9        0.24324289839890723        91.3104659421662        2.5239883678441823
10.        0.23389098927180288        91.11414422857885        2.5218556670096155
10.1        0.22951707742332056        90.97555682464845        2.521529101676001
10.200000000000001        0.2253704238781475        90.8455137010449        2.521215107977974
10.3        0.22072173732126235        90.6500379668096        2.5207328664463207
10.4        0.21623384233761417        90.52163434349232        2.5195794889395033
10.5        0.21235499571827085        90.40472564656586        2.519306545505281""".split('\n')])

#t[s]	C[10^52/(MeV s)]	E0[MeV]	beta
    M40par = np.array([np.fromstring(f,dtype = np.float, count = 4, sep = '\t') \
    for f in """0.1        0.18356234297932772        59.225788057998315        2.111893326954856
0.1	0.6353432596108131	73.08660870147284	2.3275296098414566
0.2	1.0463015461955407	92.67808980008974	2.1279233688774037
0.30000000000000004	1.5100900207234378	106.65729701934445	2.0834378941074894
0.4	1.9501281506240964	116.49892232157866	2.08067484280166
0.5	2.3886221391747324	124.3133708533452	2.08797794810306
0.6	2.790922825016633	130.7862336293636	2.0926997360711197
0.7000000000000001	3.198937904828838	136.4821231018059	2.0997244040010186
0.8	3.641033796372842	142.30293656248426	2.1044335998989467
0.9	4.144644549518473	149.00135583036797	2.1035392127019654
1.	4.72541033694239	155.33684446748094	2.1091927341684924
1.1	5.253439154414225	160.33714235754815	2.115596638553733
1.2000000000000002	5.889348206809773	166.0456724159638	2.1232403017557533
1.3000000000000003	17.450317654950126	193.2642096616553	2.7006471748386915
1.4000000000000001	21.84896439806108	199.49287685538542	2.771949119779676
1.5000000000000002	21.991606989635898	199.889555449215	2.7752101763243577
1.6	22.092330441786263	200.083136471453	2.7783295535249675
1.7000000000000002	22.06350091308194	200.19686677835614	2.780923894950702
1.8000000000000003	22.101842299171928	200.3470990679301	2.7835117727702725
1.9000000000000001	22.13348165869217	200.42109395233402	2.7856994400379413
2.	22.21627922646592	200.54748093499026	2.788466645908908
2.1	22.287815326571483	200.61940525173645	2.79175380705804
2.2	22.266435701884262	200.60348406311581	2.7940303828944235
2.3000000000000003	22.240262696759387	200.5852025174956	2.796676666145803
2.4000000000000004	22.26038501947604	200.58189376273918	2.8000019985569646
2.5000000000000004	22.22152739603574	200.52561630257236	2.8024002365322627
2.6	22.186430191671146	200.46874163177228	2.8047828023037327
2.7	22.166004047358385	200.38456297363575	2.807692339850227
2.8000000000000003	22.0803381260802	200.2608895157001	2.8092195801218223
2.9000000000000004	22.02229276905361	200.19273010079434	2.811102522273119
3.0000000000000004	22.007237406361764	200.1501023416698	2.81367581652223
3.1	22.005821568976486	200.0730813593197	2.8158630304907333
3.2	21.949154745109027	199.98058460101063	2.8181117327171687
3.3000000000000003	21.844348835738803	199.81423111173928	2.8191923132325774
3.4000000000000004	21.811848072491998	199.74063855750606	2.821178662582989
3.5000000000000004	21.73351268913697	199.64240762694584	2.82154985292711
3.6	21.636987514941776	199.5206588215984	2.8217195732748914
3.7	21.555425566537235	199.43133265827095	2.8227330436128697
3.8000000000000003	21.499127890799866	199.3476110759894	2.823671724233915
3.9000000000000004	21.539910593828424	199.26249200190165	2.8258051738965024
4.	21.486656123679293	199.18619188800255	2.82669142640182
4.1	21.4160234710523	199.11065747635442	2.827134520159014
4.2	21.349585463486193	199.0352745586242	2.8279005340772096
4.3	21.205905947690777	198.98624002082576	2.827417100841839
4.3999999999999995	21.134002503315404	198.9376737748045	2.8272528064310243
4.5	21.15243593205959	198.9066602425711	2.8280253631104877
4.6	21.102191773008496	198.82526708531296	2.82841992756189
4.7	21.059957302342276	198.76232903758637	2.828562972245873
4.8	21.005706536746175	198.69690716949847	2.8286732111303143
4.9	20.945200938277623	198.62408692892822	2.8288340775006917
5.	20.87349665110375	198.5237294455028	2.8293803633443027
5.1	20.633453881579708	198.58292541674422	2.824775726457417
5.2	20.423716092092747	198.66798266782854	2.819749231570973
5.3	20.34812369694699	198.58764767216488	2.8185999863330067
5.4	20.3113418174279	198.51137867414687	2.818869172926085
5.5	20.258185139411275	198.426815799407	2.8186289526283335
5.6	20.196932874148256	198.32439619715277	2.818621549537856
5.7	20.150590495834706	198.24043678280827	2.8191259964596767
5.8	20.037138644555743	198.15318578624064	2.8190551185968116
5.9	20.043467985236667	198.02167185284725	2.8217040319072533
6.	20.004546580601748	197.88860015774785	2.8239179878831053
6.1	19.943294162397457	197.806174214765	2.8240985514703736
6.2	19.940519060250978	197.76263559016562	2.8251213585309802
6.3	19.865661669190626	197.69664526693788	2.825417815146604
6.4	19.7025017529566	197.65286783797922	2.8238480940579582
6.5	19.56055905361094	197.61646186957088	2.822350338033718
6.6	19.517158067303725	197.570518994472	2.82229787650058
6.7	19.487531710646646	197.5059415775819	2.823069771058261
6.8	19.49293257264068	197.42647735805906	2.8241836753856706
6.9	19.479345172152712	197.34867487948333	2.824901335333801
7.	19.432924555829022	197.28959872945146	2.8255024794081245
7.1	19.365509118843637	197.23346052962663	2.8255443592673513
7.2	19.317074697615414	197.18271267149126	2.8259005395105645
7.3	19.266472083955847	197.12610939254446	2.826411859106279
7.4	19.24150150730479	197.0579035541434	2.8272387528730025
7.5	19.22138438549794	196.9769239468989	2.828058396541356
7.6	19.156945284820978	196.90477293120043	2.82862777105614
7.7	19.094346512536966	196.84184785535564	2.829054941595905
7.8	19.033494977773753	196.777857886169	2.829314883651565
7.9	19.013108640604713	196.70046012613938	2.8302800772678314
8.	19.012965239581604	196.62795300613652	2.831245492210478
8.1	18.947261611704338	196.54771855240404	2.8317196542632312
8.2	18.899859333037945	196.46981174486072	2.8323332807198365
8.3	18.879423694788265	196.39618151622076	2.833435935149982
8.4	18.840242637668435	196.31858398400473	2.8340650089088615
8.5	18.785336314089175	196.23771183906547	2.8344667516630846
8.6	18.731020787704765	196.1683807078655	2.834787316687743
8.7	18.737536733706385	196.0861058610479	2.8360353981303543
8.8	18.714322987755494	196.0022943427981	2.836773208866464
8.9	18.65488286506479	195.92567728357574	2.837231880851844
9.	18.611211722031452	195.8417850862116	2.8379294129526276
9.1	18.54403312844806	195.75267900207987	2.8379741063728416
9.2	18.488477720844937	195.67361531959594	2.838245248894724
9.3	18.43805659011539	195.5892042131494	2.8386095227700214
9.4	18.40098430667677	195.50137789816185	2.8394415001768816
9.5	18.401763805005174	195.45894260751388	2.8409674933781
9.6	18.411452761378737	195.3521494795738	2.8423707819991058
9.700000000000001	18.37659027368073	195.28102313830058	2.842842415608142
9.8	18.33948444252651	195.21600887463035	2.843338774196516
9.9	18.315554687741095	195.11995918385895	2.8441823217780624
10.	18.27147357645104	195.0258451398701	2.844694689297711
10.1	18.223690835489545	194.9462782313308	2.84513169084876
10.200000000000001	18.17133718206993	194.86826781255766	2.845603326535893
10.3	18.11984955562162	194.77853361812117	2.8457867955997997
10.4	18.074947455067473	194.68983149317316	2.846331326955668
10.5	18.03400886427662	194.6129872414095	2.8470120781426496
10.6	17.971581622531623	194.52609858904106	2.847171744185297
10.700000000000001	17.924135976965594	194.42441505053222	2.84771674777059
10.8	17.893987857406955	194.32955736544042	2.8484655717498715
10.9	17.87327961326584	194.24143944786405	2.8489003785486107
11.	17.741206231783465	194.2150811476219	2.8488788006131895
11.1	17.699967065105593	194.12983508325146	2.849700136564876
11.200000000000001	17.658626369708635	194.05778499489097	2.8505349865004352
11.3	17.619313856719973	193.9829808348378	2.8512785287867164
11.4	17.464740081345624	193.88554949284793	2.850637256272722
11.5	17.408709792387494	193.8067551997647	2.8513061621918183
11.6	17.392083083074134	193.73132496556553	2.8522414694647074
11.700000000000001	17.40553785321976	193.67913244867503	2.853928277869273
11.8	17.386786468547776	193.593998061118	2.855042929365824
11.9	17.347468315073183	193.4988969623764	2.856015056701146
12.	17.321764822695823	193.39439896497467	2.8570350974273486
12.1	17.28713075848193	193.30266106712205	2.8579596597618386
12.200000000000001	17.253169723303873	193.1825564369544	2.85875774376776
12.3	17.219061403662213	193.08295638277067	2.8596699113532242
12.4	17.178921735884423	192.9892276268714	2.860489298270473
12.5	17.139051562328138	192.89969933033453	2.861134541561836
12.6	17.09878544221813	192.80435205969528	2.8618997925384155
12.700000000000001	17.107930759577883	192.73112638851748	2.8636312354379903
12.8	17.056172825204275	192.6425767751638	2.864281522134938
12.9	17.02291323238282	192.51700516158976	2.8650067793875236
13.	16.98344934849609	192.43444948500166	2.865737398751351
13.1	16.923049607913583	192.33359309921386	2.8663202929117455
13.200000000000001	16.872823603060034	192.22789174118023	2.867026932554831
13.3	16.84073292356619	192.1397331826696	2.8678134654834437
13.4	16.802161189214893	192.04705111182645	2.8682968671760136""".split('\n')])

    def __init__(self, Mprog):
        """
        Initialialize the class

        Parameters
        ----------
        Mprog: float
            the progenitor mass in solar masses. At the moment, only 10 and 18 solar masses
            are implmented
        """

        if not (Mprog == 10. or Mprog == 18. or Mprog == 40.):
            raise ValueError("Mprog must be either 10. or 18. or 40. solar masses")

        self._Mprog = Mprog

        self.__set_interpolation()
        return

    @property
    def Mprog(self):
        return self._Mprog

    @Mprog.setter
    def Mprog(self, Mprog):
        if not (Mprog == 10. or Mprog == 18.):
            raise ValueError("Mprog must be either 10. or 18. solar masses")
        self._Mprog = Mprog
        self.__set_interpolation()
        return 

    def __set_interpolation(self):
        """Set the interpolation tables for spectrum and light curve"""
        self.__spls = []
        if self._Mprog == 10.:
            Mpar = ALPSNSignal.M10par
        elif self._Mprog == 18.:
            Mpar = ALPSNSignal.M18par
        elif self._Mprog == 40.:
            Mpar = ALPSNSignal.M40par
        for i in range(1,Mpar.shape[1]):
            self.__spls.append(USpline(Mpar[:,0],
                    Mpar[:,i],
                    s = 0, k = 1, ext = 'extrapolate'))
        return

    def dnde(self,g10, EMeV, t):
        return g10**2. * self.__spls[0](t) * (EMeV/self.__spls[1](t)) ** self.__spls[2](t) \
                * np.exp(-(self.__spls[2](t) + 1.) * EMeV / self.__spls[1](t))  

    def __call__(self,EMeV, ts, g10 = 1.):
        """
        Calculate the ALP flux for a given energy and time
        in units of 10^52/MeV/s

        Parameters
        ----------
        EMeV: float or `~numpy.ndarray`
            Energies in MeV
        ts: float or `~numpy.ndarray`
            time in seconds

        kwargs
        ------
        g10: float
            Photon-ALP coupling in 10^-10 / GeV

        Returns
        -------
        squeezed `~numpy.ndarray` with ALP flux for each energy and 
        time
        """
        if np.isscalar(EMeV):
            EMeV = np.array([EMeV])
        if np.isscalar(ts):
            ts = np.array([ts])
            
        ee,tt = np.meshgrid(EMeV, ts, indexing = 'ij')
        f = self.dnde(g10, ee,tt)
        f[f < 0.] = np.zeros(np.sum(f < 0.))
        return np.squeeze(f)

if __name__ == '__main__':
    usage = "usage: %(prog)s sntable [options]"
    description = "Generate ALP signals for given extragal. supernovae"
    parser = argparse.ArgumentParser(usage=usage,description=description)
    parser.add_argument('sntable', help = 'fits file with SNe', nargs="+")
    parser.add_argument('--Mprog', default = 10., type=float,
                                    choices = [10.,18.],
                                    help = 'Progenitor mass in solar masses')
    parser.add_argument('--mstart', default = 0.1, type=float,
                                    help = 'minimum tested ALP mass in neV')
    parser.add_argument('--mstop', default = 100., type=float,
                                    help = 'maximum tested ALP mass in neV')
    parser.add_argument('--mstep', default = 13, type=int,
                                help = "number of tested ALP masses")
    parser.add_argument('--Estart', default = 10., type=float,
                                    help = 'minimum Energy in MeV')
    parser.add_argument('--Estop', default = 2000., type=float,
                                    help = 'maximum Energy in MeV')
    parser.add_argument('--Estep', default = 203, type=int,
                                help = "number of Energies")
    parser.add_argument('--tstart', default = 0.1, type=float,
                                    help = 'minimum time in s')
    parser.add_argument('--tstop', default = 20., type=float,
                                    help = 'maximum time in s')
    parser.add_argument('--tstep', default = 201, type=int,
                                help = "number of time steps")
    parser.add_argument('--gref', default = 1., type=float,
                                    help = 'photon-ALP coupling for which calculations '+\
                                        'will be performed, in 10^-11 / GeV')
    args = parser.parse_args()

    # set the arrays
    asn = ALPSNSignal(args.Mprog)
    EMeV = np.logspace(np.log10(args.Estart),np.log10(args.Estop),args.Estep)
    ts = np.logspace(np.log10(args.tstart),np.log10(args.tstop),args.tstep)
    mneV = np.logspace(np.log10(args.mstart),np.log10(args.mstop),args.mstep)

    # get the SN table
    tsn = Table.read(args.sntable[0])
    if len(args.sntable) > 1:
        for i in range(1,len(args.sntable)):
            tsn = vstack((tsn,Table.read(args.sntable[i])))

    # ALP flux in 1/MeV/s
    # shape after transposure is t-dim x E-dim
    f = asn(EMeV,ts,g10 = args.gref / 10.).T * 1e52

    # initial polarization
    pin = np.diag((0.,0.,1.))

    # cosmology
    cosmo = FlatLambdaCDM(H0 = 70., Om0 = 0.3)

    # loop over the sntable and ALP masses
    c = OrderedDict() # columns for new table

    c['name'] = tsn['name'].data
    c['time'] = []

    for t in tsn:
        src = Source(z = float(t['z']),
                    ra = float(t['ra']),
                    dec = float(t['dec']))
        mod = ModuleList(ALP(m=1.,g=args.gref), src, pin = pin, EGeV = EMeV / 1e3)
        mod.add_propagation("GMF",0, model = 'jansson12', model_sum = 'ASS')
        c['time'].append(ts)

        for i,m in enumerate(mneV):
            if not '{0:.3f}'.format(m) in c.keys():
                c['{0:.3f}'.format(m)] = []

            mod.alp.m = m

            px,py,pa = mod.run(multiprocess=2)

            # calculate photon flux in photons / MeV / s / cm^2:
            d = cosmo.luminosity_distance(t['z'])
            flux = f * (px[0] + py[0]) / 4. / np.pi / d.to('cm') ** 2.

            # integrate flux over energy to obtain light curve in photons / s / cm^2
            flux_int = simps(flux * EMeV, np.log(EMeV), axis = 1)
            c['{0:.3f}'.format(m)].append(flux_int)

        del mod

    result = Table(c)
    result.write('ALPSNSignal_Mprog{0:.0f}.fits'.format(asn.Mprog),
                overwrite = True)
