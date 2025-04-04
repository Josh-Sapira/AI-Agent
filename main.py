from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# Initialize model
model = OllamaLLM(model="llama3.2")

# Set up question template
template = """
You are an expert in Major League Baseball season stats for players from the 2023 season.

The data you are receiving includes encoded information about their stats, depending on whether the player is a position player or pitcher.

The pitcher stats are as follows:
Rk : Rank
Name : Player name
Age : Player's age
Tm : Team
Lg : League
W : Wins
L : Losses
W-L% : Win-Loss percentage
ERA : 9 * ER / IP
G : Games played
GS : Games started
GF : Games finished
CG : Complete game
SHO : Shutouts
SV : Saves
IP : Innings pitched
H : Hits/Hits allowed
R : Runs scored/allowed
ER : Earned runs allowed
HR : Home runs hit/allowed
BB : Bases on balls/walks
IBB : Intentional bases on balls
SO : Strikeouts
HBP : Times hit by a pitch
BK : Balks
WP : Wild pitches
BF : Batters faced
ERA+ : 100 * (logERA/ERA)
FIP : Fielding independent pitching. Measures a pitcher's effectiveness at HR, BB, HBP and causing SO.
WHIP : (BB + H) / IP
H9 : 9 * H / IP
HR9 : 9 * HR / IP
BB9 : 9 * BB / IP
SO9 : 9 * SO / IP
SO/W : SO / W

The hitter stats are as follows:
Rk : Rank
Name : Player name
Age : Player's age
Tm : Team
Lg : League
G : Games played
PA : Plate appearances
AB : At bats
R : Runs scored/allowed
H : Hits/hits allowed
2B : Doubles hit/allowed
3B : Triples hit/allowed
HR : Home runs hit/allowed
RBI : Runs batted in
SB : Stolen bases
CS : Caught stealing
BB : Bases on balls/walks
SO : Strikeouts
BA : Hits/at bats
OBP : (H + BB + HBP) / (AB + BB + HBP + SF)
SLG : Total bases/at bats or (1B + 2 * 2B + 3 * 3B + 4 * HR) / AB
OPS : On-base + Slugging percentages
OPS+ : 100 * (OBP / logOBP + SLG / logSLG - 1)
TB : Total bases
GDP : Double plays grounded into
HBP : Times hit by a pitch
SH : Sacrifice hits
SF : Sacrifice flies
IBB : Intentional bases on balls

Here is the player data: {players}

Here is the question to answer: {question}
"""

# Set up chain to provide prompt from template to model
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Loop to continually prompt from user
while True:
    print("\n\n----------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")

    # Stop asking for questions if user quits
    if question == 'q':
        break

    # Get relevant data from retriever
    players = retriever.invoke(question)

    # Print result of the chain
    result = chain.invoke({"players": players, "question":question})
    print(result)