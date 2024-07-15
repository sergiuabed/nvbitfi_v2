import os, sys, shutil



def main():
    os.chdir("./logs")
    path=os.getcwd()
    directories=os.listdir()
    instructions={}
    for each_directory in directories:
        profile_file=path+"/"+each_directory+"/nvbitfi-igprofile.txt"
        if(os.path.isfile(profile_file) and "S" in each_directory):            
            #print(profile_file)
            if(each_directory not in instructions):
                instructions[each_directory]={}                
            try:
                with open(profile_file) as profile:
                    for line in profile:
                        #if "NS_srad_v1" in each_directory: print(line)
                        InstrKey=((line.strip()).split(';')[4]).strip().split(':')[0]
                        Instrval=int(((line.strip()).split(';')[4]).strip().split(':')[1])
                        if(InstrKey not in instructions[each_directory] and InstrKey!=""):
                            instructions[each_directory][InstrKey]=int(Instrval)
                        else:
                            instructions[each_directory][InstrKey]+=int(Instrval)

                        ctasKey=((line.strip()).split(';')[3]).strip().split(':')[0]
                        ctasval=int(((line.strip()).split(';')[3]).strip().split(':')[1])
                        if(ctasKey not in instructions[each_directory] and ctasKey!=""):
                            instructions[each_directory][ctasKey]=int(ctasval)
                        else:
                            instructions[each_directory][ctasKey]+=int(ctasval)

                        threadsKey=((line.strip()).split(';')[6]).strip().split(':')[0]
                        threadsval=int(((line.strip()).split(';')[6]).strip().split(':')[1].replace(",",""))
                        if(threadsKey not in instructions[each_directory] and threadsKey!=""):
                            instructions[each_directory][threadsKey]=int(threadsval)
                        else:
                            instructions[each_directory][threadsKey]+=int(threadsval)

                        Instr=((line.strip()).split(';')[5]).strip().split(',')
                        for each_inst in Instr:
                            if each_inst!="":
                                key=each_inst.lstrip().split(':')[0]
                                val=each_inst.lstrip().split(':')[1]
                                if(key not in instructions[each_directory] and key!=""):
                                    instructions[each_directory][key]=int(val)
                                else:
                                    instructions[each_directory][key]+=int(val)

                        
                    profile.close()

            except OSError as err:
                print("OS error:", err)
            except ValueError:
                print("Error ")
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise
            
    with open('report.csv','w') as report:
        instr=[key for key in instructions["NS_NN"]]
        first_line="App"
        for inst in instr:
            first_line+=f", {inst}"
        report.write(first_line+'\n')
        for appname in instructions:
            first_line=appname
            for inst in instr:
                first_line+=f", {instructions[appname][inst]}"
            report.write(first_line+'\n')
            

if __name__=="__main__":
    main()