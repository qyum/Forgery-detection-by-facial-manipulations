import os
import cv2
from skimage.measure import compare_ssim
import numpy as np
 
names=['vudstovrck', 'jdubbvfswz', 'atvmxvwyns', 'qzimuostzz', 'kbvibjhfzo', 'ccfoszqabv', 'fjlyaizcwc', 'ffcwhpnpuw', 'slwkmefgde', 'fysyrqfguw', 'qjlhemtkxk', 'dlpoieqvfb', 'qzimuostzz', 'proiippuup', 'gxembgiarp', 'iufotyxgzb', 'aytzyidmgs', 'dkuayagnmc', 'jomvcqqars', 'sunqwnmlkx', 'ygdgwyqyut', 'bzythlfnhq', 'lyvlnqduqg', 'yxyhvdlrgk', 'swedbyuehz', 'xlbnmndmku', 'meawmsgiti', 'ppdpgwyjgm', 'fewcljwqkr', 'cppdvdejkc', 'tqhbgzfwsf', 'meawmsgiti', 'vcxckqbaya', 'xobhsemxmv', 'xwcggrygwl', 'fysyrqfguw', 'xxsxktyvzt', 'bulkxhhknf', 'edyncaijwx', 'jkddywriuf', 'uonshkejav', 'xngpzquyhs', 'fjlyaizcwc', 'meawmsgiti', 'inkqxytzyu', 'znjupdqnwo', 'jepguaulgf', 'dzyuwjkjui', 'upgerjvcjb', 'wwqiuiwdbz', 'mfzqxktxud', 'qedsgieuqn', 'ppdpgwyjgm', 'ifbdbogiqn', 'mfzqxktxud', 'hyhjfdxqxy', 'jdubbvfswz', 'efwfxwwlbw', 'gktjowiuqe', 'qzklcjjxdq', 'gbqrgajyca', 'rrcsuwgpnd', 'xclqbefnvc', 'cprhtltsjp', 'ujzwwfkeia', 'meawmsgiti', 'topyiohccg', 'mfnowqfdwl', 'rvoudrbyac', 'xxrzzncksa', 'jjyfvzxwwx', 'djxdyjopjd', 'xnhcreiyqg', 'imzqmbfugn', 'cpjxareypw', 'fysyrqfguw', 'iyefnuagav', 'ybjrqnqnno', 'qtnjyomzwo', 'keecvpbncd', 'fecysfujzk', 'kysxawkest', 'xzvrgckqkz', 'gxembgiarp', 'sgjnvxvcpu', 'woshnzbxmc', 'znpdbbsfvj', 'rlvgtsjyer', 'vcxckqbaya', 'yfsnwkbafm', 'kydlpqfrvv', 'tdohqkzvbk', 'yfsnwkbafm', 'mfpgdgsaxg', 'hyuipchisa', 'ybjrqnqnno', 'gjypopglvi', 'xqnykluhws', 'dbtbbhakdv', 'rrcsuwgpnd', 'ptokilxwcx', 'gneufaypol', 'vmospzljws', 'mmhqllmlew', 'fntskqfxxf', 'xnfwdpptym', 'ellavthztb', 'ytufbmkdlq', 'dbtbbhakdv', 'sfujxhuyje', 'oesxbvktem', 'bxzakyopjf', 'lmlyvmfbbe', 'cizlkenljw', 'ytufbmkdlq', 'iklzfeueid', 'meawmsgiti', 'atvmxvwyns', 'xagsvjctmp', 'djxdyjopjd', 'yagllixjvh', 'uuxqylnzls', 'ywvlvpvroj', 'fdcttsvjwf', 'ubplsigbvj', 'brwrlczjvi', 'gxhcuxulhi', 'drcyabprvt', 'xwcggrygwl', 'crezycjqyk', 'cyxlcuyznd', 'kgbkktcjxf', 'mgowkzsbyx', 'fhghkqdkhe', 'mmhqllmlew', 'ekcrtigpab', 'ywvlvpvroj', 'xzvrgckqkz', 'gktjowiuqe', 'gipbyjfxfp', 'liniegczcx', 'ifjktxxiln', 'kgbkktcjxf', 'nvpluswotp', 'fdcttsvjwf', 'sqwvfgwdxr', 'vgqotmftcr', 'lietldeotq', 'svcnlasmeh', 'gxhcuxulhi', 'itzmdwutdu', 'xngpzquyhs', 'ehtdtkmmli', 'atvmxvwyns', 'euqpvnyxrb', 'kysxawkest', 'gjypopglvi', 'fygviyzcjm', 'ztbinwxgyu', 'tfoxelmnjx', 'qypgyrxcme', 'puppdcffcj', 'topyiohccg', 'zrkinjhsuq', 'bwipwzzxxu', 'atkdltyyen', 'qzimuostzz', 'ztbinwxgyu', 'kwyvikrgmx', 'bwipwzzxxu', 'jomvcqqars', 'fpvduejzcw', 'tfoixxmpoo', 'swedbyuehz', 'vrsinxahfh', 'wgmbcqfgkp', 'vrsinxahfh', 'iklzfeueid', 'fkyrrigzpt', 'lkdlzpkukw', 'chtapglbcj', 'iiomvouemm', 'gdfyzwykty', 'fecysfujzk', 'qeumxirsme', 'kgbkktcjxf', 'wtreibcmgm', 'kgbkktcjxf', 'qtnjyomzwo', 'dlpoieqvfb', 'qapnbtdypb', 'qeumxirsme', 'atvmxvwyns', 'gbqrgajyca', 'caifxvsozs', 'meawmsgiti', 'atvmxvwyns', 'kdodrvufdh', 'atvmxvwyns', 'qtnjyomzwo', 'ffcwhpnpuw', 'jjyfvzxwwx', 'yxyhvdlrgk', 'ellavthztb', 'iuzdfwsefw', 'wapebjxejr', 'luvasmspox', 'qtnjyomzwo', 'ijokcwewbs', 'qokxxuayqn', 'ptokilxwcx', 'ljaifbsfuw', 'ywvlvpvroj', 'bffwsjxghk', 'jzmdganfys', 'duycddgtrl', 'jszyyhamrh', 'uonshkejav', 'cppdvdejkc', 'tdohqkzvbk', 'ehccixxzoe', 'gbqrgajyca', 'gneufaypol', 'fgfyrfyqay', 'gmihbscmwq', 'smggzgxymo', 'jljpdojupu', 'zhfyuhonra', 'mmhqllmlew', 'qeumxirsme', 'hcswybumab', 'rfzzrftgco', 'ztbinwxgyu', 'ezaajaswoe', 'tamayudqqx', 'gnyspcpbhd', 'eckvhdusax', 'fkqptfouqw', 'xnfwdpptym', 'ngdswpaqnt', 'kkzsnmrkqk', 'iiomvouemm', 'qeumxirsme', 'zrkinjhsuq', 'iyefnuagav', 'kydlpqfrvv', 'rlldzrnmdn', 'yagllixjvh', 'jomvcqqars', 'dzyuwjkjui', 'dzyuwjkjui', 'nlerwupaqr', 'ohnonevlro', 'gjypopglvi', 'dakiztgtnw', 'gzyzdcbuuv', 'iieoqptzec', 'fgfyrfyqay', 'fgfyrfyqay', 'slwkmefgde', 'kuelhabsmz', 'tmdformfqp', 'imzqmbfugn', 'yoavqsqobz', 'pylnolwenx', 'xwcggrygwl', 'bgwmmujlmc', 'pqvypayzrp', 'lyvlnqduqg', 'yjlsxqoauz', 'ehccixxzoe', 'iufotyxgzb', 'fysyrqfguw', 'kgbkktcjxf', 'grnycmbdfu', 'kysxawkest', 'xngpzquyhs', 'itmwoyxbas', 'iieoqptzec', 'rjlgchzmfv', 'qzklcjjxdq', 'qzklcjjxdq', 'olakcrnuro', 'abarnvbtwb', 'lyvlnqduqg', 'xngpzquyhs', 'hqtepxaeqx', 'gipbyjfxfp', 'bejhvclboh', 'nzquxipbye', 'avmjormvsx', 'sasoxcqisz', 'ixuouyigxa', 'joeifeskbs', 'xzvrgckqkz', 'ywvlvpvroj', 'jwcsqxzdlv', 'egghxjjmfg', 'tivkmbqgwp', 'gipbyjfxfp', 'qhkqqfznrg', 'dzyuwjkjui', 'cprhtltsjp', 'ybjrqnqnno', 'lulmevqtla', 'gbqrgajyca', 'kuelhabsmz', 'svcnlasmeh', 'dbnygxtwek', 'cmbzllswnl', 'nvpluswotp', 'grnycmbdfu', 'gdfyzwykty', 'rrcsuwgpnd', 'ybetenmsye', 'gomwfvijiv', 'qeumxirsme', 'qzklcjjxdq', 'gipbyjfxfp', 'wtreibcmgm', 'afoovlsmtx', 'bdnaqemxmr', 'gjypopglvi']
names_1=['aagfhgtpmv', 'aapnvogymq', 'abofeumbvv', 'abqwwspghj', 'acifjvzvpm', 'acqfdwsrhi', 'acxnxvbsxk', 'acxwigylke', 'aczrgyricp', 'adhsbajydo', 'adohikbdaz', 'adylbeequz', 'aelzhcnwgf', 'aettqgevhz', 'aevrfsexku', 'agdkmztvby', 'agqphdxmwt', 'ahbweevwpv', 'ahdbuwqxit', 'ahfazfbntc', 'aipfdnwpoo', 'ajwpjhrbcv', 'aklqzsddfl', 'aknbdpmgua', 'aknmpoonls', 'akvmwkdyuv', 'akxoopqjqz', 'akzbnazxtz', 'aladcziidp', 'alaijyygdv', 'alninxcyhg', 'altziddtxi', 'alvgwypubw', 'amaivqofda', 'amowujxmzc', 'andaxzscny', 'aneclqfpbt', 'aorjvbyxhw', 'apatcsqejh', 'apgjqzkoma', 'apogckdfrz', 'aqpnvjhuzw', 'arkroixhey', 'arlmiizoob', 'arrhsnjqku', 'asdpeebotb', 'aslsvlvpth', 'asmpfjfzif', 'asvcrfdpnq', 'atxvxouljq', 'atyntldecu', 'atzdznmder', 'aufmsmnoye', 'augtsuxpzc', 'avfitoutyn', 'avgiuextiz', 'avibnnhwhp', 'avnqydkqjj', 'avssvvsdhz', 'avtycwsgyb', 'avvdgsennp', 'avywawptfc', 'awhmfnnjih', 'awnwkrqibf', 'awukslzjra', 'axczxisdtb', 'axoygtekut', 'axwgcsyphv', 'axwovszumc', 'ayqvfdhslr', 'azpuxunqyo', 'azsmewqghg', 'bahdpoesir', 'bbhpvrmbse', 'bbhtdfuqxq', 'bbvgxeczei', 'bchnbulevv', 'bctvsmddgq', 'bdbhekrrwo', 'bdgipnyobr', 'bdxuhamuqx', 'benmsfzfaz', 'bgaogsjehq', 'bggsurpgpr', 'bghphrsfxf', 'bgmlwsoamc', 'bguwlyazau', 'bhaaboftbc', 'bhbdugnurr', 'bhpwpydzpo', 'bhsluedavd', 'bjjbwsqjir', 'bjkmjilrxp', 'bjsmaqefoi', 'bkmdzhfzfh', 'bkvetcojbt', 'bkwxhglwct', 'blpchvmhxx', 'blzydqdfem', 'bmbbkwmxqj', 'bmehkyanbj', 'bmhvktyiwp', 'bmioepcpsx', 'bmjmjmbglm', 'bnbuonyoje', 'bndybcqhfr', 'bnjcdrfuov', 'bntlodcfeg', 'bofqajtwve', 'boovltmuwi', 'bopqhhalml', 'bourlmzsio', 'bpwzipqtxf', 'bpxckdzddv', 'bqdjzqhcft', 'bqeiblbxtl', 'bqhtpqmmqp', 'bqkdbcqjvb', 'bqnymlsayl', 'bqqpbzjgup', 'bqtuuwzdtr', 'brhalypwoo', 'brvqtabyxj', 'bseamdrpbj', 'bsfmwclnqy', 'bsqgziaylx', 'btiysiskpf', 'btjlfpzbdu', 'btjwbtsgln', 'btmsngnqhv', 'btohlidmru', 'btugrnoton', 'btunxncpjh', 'btxlttbpkj', 'bvgwelbeof', 'bvzjkezkms', 'bweezhfpzp', 'bwuwstvsbw', 'bydaidkpdp', 'byfenovjnf', 'byijojkdba', 'byofowlkki', 'byqzyxifza', 'byunigvnay', 'byyqectxqa', 'bzmdrafeex', 'caqbrkogkb', 'cbbibzcoih', 'cbltdtxglo', 'ccmonzqfrz', 'cdaxixbosp', 'cdbsbdymzd', 'cdphtzqrvp', 'cdyakrxkia', 'cepxysienc', 'cettndmvzl', 'ceymbecxnj', 'cferslmfwh', 'cffffbcywc', 'cfyduhpbps', 'cglxirfaey', 'cgvrgibpfo', 'chzieimrwu', 'ckbdwedgmc', 'cknyxaqouy', 'cksanfsjhc', 'clihsshdkq', 'cmxcfkrjiv', 'cnilkgvfei', 'coadfnerlk', 'covdcysmbi', 'cqfugiqupm', 'cqhngvpgyi', 'cqrskwiqng', 'crktehraph', 'crzfebnfgb', 'cthdnahrkh', 'ctpqeykqdp', 'cttqtsjvgn', 'ctzmavwror', 'curpwogllm', 'cuzrgrbvil', 'cvaksbpssm', 'cwbacdwrzo', 'cwqlvzefpg', 'cwrtyzndpx', 'cwsbspfzck', 'cwwandrkus', 'cxfujlvsuw', 'cxrfacemmq', 'cxttmymlbn', 'cyboodqqyr', 'cycacemkmt', 'cyclgfjdrv', 'czfunozvwp', 'czkdanyadc', 'czmqpxrqoh', 'dafhtipaml', 'dakqwktlbi', 'dbhoxkblzx', 'dbhrpizyeq', 'dboxtiehng', 'dbzcqmxzaj', 'dbzpcjntve', 'dcamvmuors', 'dcuiiorugd', 'ddhfabwpuz', 'ddjggcasdw', 'ddpvuimigj', 'ddqccgmtka', 'degpbqvcay', 'deywhkarol', 'deyyistcrd', 'dfbpceeaox', 'dgmevclvzy', 'dgxrqjdomn', 'dgzklxjmix', 'dhcselezer', 'dhevettufk', 'dhjmzhrcav', 'dhkwmjxwrn', 'dhoqofwoxa', 'diomeixhrg', 'diopzaywor', 'diqraixiov', 'diuzrpqjli', 'djvtbgwdcc', 'djvutyvaio', 'dkdwxmtpuo', 'dkhlttuvmx', 'dkrvorliqc', 'dkwjwbwgey', 'dlrsbscitn', 'dnexlwbcxq', 'dnhvalzvrt', 'dntkzzzcdh', 'dnyvfblxpm', 'doanjploai', 'dofusvhnib', 'dozyddhild', 'dptbnjnkdg', 'dptrzdvwpg', 'dqnyszdong', 'dqppxmoqdl', 'dqqtjcryjv', 'dqswpjoepo', 'dqzreruvje', 'drgjzlxzxj', 'drsakwyvqv', 'drtbksnpol', 'dsdoseflas', 'dsgpbgsrdm', 'dsndhujjjb', 'dtbpmdqvao', 'dtocdfbwca', 'dubiroskqn', 'dulanfulol', 'duvyaxbzvp', 'duzuusuajr', 'dvakowbgbt', 'dvumqqhoac', 'dwediigjit', 'dxuliowugt', 'dxuplhwvig', 'dzieklokdr', 'dzqwgqewhu', 'dzvyfiarrq', 'dzwkmcwkwl', 'eahlqmfvtj', 'eajlrktemq', 'ebchwmwayp', 'ebebgmtlcu', 'ebeknhudxq', 'ebkzwjgjhq', 'ebywfrmhtd', 'ecnihjlfyt', 'ecuvtoltue', 'ecwaxgutkc', 'eczrseixwq', 'eebrkicpry', 'eebserckhh', 'eejswgycjc', 'eekozbeafq', 'eepezmygaq', 'eeyhxisdfh', 'efdyrflcpg', 'egbbcxcuqy', 'ehbnclaukr', 'ehdkmxgtxh', 'ehevsxtecd', 'ehfiekigla', 'ehieahnhte', 'eiriyukqqy', 'eivxffliio', 'eiwopxzjfn', 'eixwxvxbbn', 'ejkqesyvam', 'ekhacizpah', 'ekkdjkirzq', 'elginszwtk', 'elvvackpjh', 'emaalmsonj', 'emfbhytfhc', 'emgjphonqb', 'ensyyivobf', 'eoewqcpbgt', 'eprybmbpba', 'epymyyiblu', 'eqjscdagiv', 'eqvuznuwsa', 'erqgqacbqe', 'errocgcham', 'esckbnkkvb', 'esgftaficx', 'esnntzzajv', 'esxrvsgpvb', 'esyhwdfnxs', 'esyrimvzsa', 'etdcqxabww', 'etejaapnxh', 'etmcruaihe', 'etohcvnzbj', 'eukvucdetx']

for index in range(0,len(names)):
    #if len(names[index]) > best:
    #best = names_1[index]
    root_dir='F:/deepfake_data/train_sample_videos_2/' 
    ori_id=os.path.join(root_dir,"{}.mp4".format(names[index]))
    #print(best)
    capture_ori = cv2.VideoCapture(ori_id)
    frames_num_ori = int(capture_ori.get(cv2.CAP_PROP_FRAME_COUNT))
    #print(frames_num_ori)
    fake_id=os.path.join(root_dir,"{}.mp4".format(names_1[index]))
    #print(best_1)
    capture_fake = cv2.VideoCapture(fake_id)
    frames_num_fake = int(capture_fake.get(cv2.CAP_PROP_FRAME_COUNT))
    #print(frames_num_ori)
    for j in range(max(frames_num_ori,frames_num_fake)):
            
            #........for ori_id.............................
            capture_ori.grab()
            if j % 10 != 0:
                continue
            success, frame_1 = capture_ori.retrieve()
            
            if not success:
                continue
            #print(frame_1)
            id_1 = os.path.splitext(os.path.basename(ori_id))[0]
            
            #........for fake id...........................
            capture_fake.grab()
            if j % 10 != 0:
                continue
            success, frame_2 = capture_fake.retrieve()
            
            if not success:
                continue
            id_2= os.path.splitext(os.path.basename(fake_id))[0]
            #print(frame_2)
            
            #........ ...................SSIm differences...................................
            #try:
            d, a = compare_ssim(frame_1,frame_2, multichannel=True, full=True)
            #d,a  =  [compare_ssim(frame_1[i],frame_2[i],
            #multichannel=True, full=True) for i in range(0, max(len(frame_1), len(frame_2)))]
            a = 1 - a
            diff = (a * 255).astype(np.uint8)
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            diff_path='F:/deepfake_data/ssim_diff.jpg'
            #cv2.imwrite(diff_path, diff)
            cv2.imwrite(os.path.join(diff_path,"{}_{}_{}.jpg".format(id_1,id_2, j)),diff)
            #except:
                   #pass
