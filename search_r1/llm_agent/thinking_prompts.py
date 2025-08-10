meta_thinking_prompt = "<think> Before continuing, I needed to pause to consider whether I was on the right track. Do I already have enough info? Does my logic sound? Any conflicting facts?  I'll first choose one of Retrieve / Reflect / Check-Conflict and say it as <action> [desired action] </action>. Then I will continue my elaborate reasoning."

action_invalid_prompt = f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n'
# revision_prompts = {
#     0: "I realized I might have overlooked something important.",
#     1: "Considering my current reasoning, I might have made a misstep.",
#     2: "There's a possibility I've missed critical information or logic flaws.",
#     3: "I paused to reflect on potential errors or gaps in my thinking.",
#     4: "Upon reconsideration, my reasoning might have subtle errors.",
#     5: "It's clear I must double-check my reasoning to ensure no mistakes were made.",
#     6: "Acknowledging I could have misinterpreted some information.",
#     7: "I recognized my reasoning might not be fully robust.",
#     8: "Feeling uncertain about the accuracy of my previous step.",
#     9: "Considering I might have overlooked or misjudged some elements.",
#     10: "Recognizing potential blind spots in my analysis.",
#     11: "Since my logic might not be entirely sound.",
#     12: "Reflecting briefly, I see my earlier reasoning might contain errors.",
#     13: "I suspect something could be wrong or incomplete in my logic.",
#     14: "Given potential flaws in my current approach.",
#     15: "I need to check carefully—my earlier assumptions might be questionable.",
#     16: "Considering there may be hidden flaws or uncertainties.",
#     17: "Realizing that my reasoning might not be bulletproof.",
#     18: "To avoid mistakes, I should reconsider carefully.",
#     19: "There could be errors or gaps in my understanding.",
#     20: "Recognizing potential issues with my prior logic.",
#     21: "Given that my logic might have some weak points.",
#     22: "I need to be careful—perhaps I missed something crucial.",
#     23: "Aware that my previous thinking might be flawed or incomplete.",
#     24: "Given my earlier uncertainty or potential oversight.",
#     25: "Suspecting I may have made an error.",
#     26: "Recognizing my previous step might have issues.",
#     27: "I sense my reasoning might be off track.",
#     28: "Considering I could have missed something.",
#     29: "Realizing potential gaps in logic or information."
# }

# random_15
revision_prompts = {
    0: "<think> Perhaps I've overlooked critical points or slipped up in my logic.",
    1: "<think> There might be key gaps in my understanding or errors in reasoning.",
    2: "<think> Maybe I've misjudged something important or neglected key facts.",
    3: "<think> Reflecting now, I might have overlooked critical data or erred somewhere.",
    4: "<think> I'm sensing a gap or error might be present in my recent reasoning.",
    5: "<think> Aware that my reasoning might be flawed or lacking crucial points.",
    6: "<think> I need to reconsider—I might've skipped vital information or erred.",
    7: "<think> There's a chance my previous thinking has unnoticed mistakes or omissions.",
    8: "<think> Perhaps my earlier reasoning has hidden mistakes or missing information.",
    9: "<think> I might have unintentionally ignored essential details or misunderstood something.",
    10: "<think> Revisiting carefully, perhaps errors or oversights went unnoticed earlier.",
    11: "<think> Maybe important points slipped my attention, or I made a miscalculation.",
    12: "<think> Possibly, I misunderstood something fundamental or missed key evidence.",
    13: "<think> Concerned about potential unnoticed mistakes or overlooked essential details.",
    14: "<think> I'm doubting if crucial points were missed or mistakes made earlier.",
}


# random_5
# revision_prompts = {
#     0: "<think> I wonder if vital information escaped my notice or if I made an error.",
#     1: "<think> I need to reconsider—I might've skipped vital information or erred.",
#     2: "<think> There's a chance my previous thinking has unnoticed mistakes or omissions.",
#     3: "<think> Perhaps my earlier reasoning has hidden mistakes or missing information.",
#     4: "<think> It's conceivable that I've neglected critical information or erred.",
# }

# 原始
# revision_prompts = {
#     0: "<think> Perhaps I've overlooked critical points or slipped up in my logic.",
#     1: "<think> I wonder if vital information escaped my notice or if I made an error.",
#     2: "<think> There might be key gaps in my understanding or errors in reasoning.",
#     3: "<think> It's possible I've missed something important or misunderstood crucial details.",
#     4: "<think> I suspect errors crept in, or essential points went unnoticed.",
#     5: "<think> Maybe I've misjudged something important or neglected key facts.",
#     6: "<think> Reflecting now, I might have overlooked critical data or erred somewhere.",
#     7: "<think> Possibly, I've missed significant insights or made a mistake.",
#     8: "<think> I'm sensing a gap or error might be present in my recent reasoning.",
#     9: "<think> I could have misinterpreted important facts or overlooked necessary details.",
#     10: "<think> Aware that my reasoning might be flawed or lacking crucial points.",
#     11: "<think> I need to reconsider—I might've skipped vital information or erred.",
#     12: "<think> There's a chance my previous thinking has unnoticed mistakes or omissions.",
#     13: "<think> I feel there might be something critical I overlooked or misunderstood.",
#     14: "<think> Perhaps my earlier reasoning has hidden mistakes or missing information.",
#     15: "<think> Concerned I might have overlooked key aspects or made subtle errors.",
#     16: "<think> I might have unintentionally ignored essential details or misunderstood something.",
#     17: "<think> Revisiting carefully, perhaps errors or oversights went unnoticed earlier.",
#     18: "<think> Maybe important points slipped my attention, or I made a miscalculation.",
#     19: "<think> It's likely I've overlooked something crucial or stumbled in logic.",
#     20: "<think> Reflecting, I could've missed critical clues or made errors in judgment.",
#     21: "<think> Possibly, I misunderstood something fundamental or missed key evidence.",
#     22: "<think> Concerned about potential unnoticed mistakes or overlooked essential details.",
#     23: "<think> Perhaps my earlier step wasn't entirely accurate or lacked vital points.",
#     24: "<think> It's conceivable that I've neglected critical information or erred.",
#     25: "<think> Wondering if I've mistakenly dismissed something important or misunderstood it.",
#     26: "<think> Maybe my previous reasoning has blind spots or unnoticed errors.",
#     27: "<think> I'm doubting if crucial points were missed or mistakes made earlier.",
#     28: "<think> Feeling uncertain—perhaps critical details slipped past or were misunderstood.",
#     29: "<think> Recognizing possible gaps or missteps I didn't previously notice."
# }

# o3_rephrase_30
# revision_prompts = {
#     0: "<think> Maybe I overlooked crucial points or my reasoning went off track.",
#     1: "<think> I wonder if key facts escaped me or I erred in my logic.",
#     2: "<think> There could be missing pieces I ignored or flaws in my argument.",
#     3: "<think> It's possible I missed something vital or misread crucial details.",
#     4: "<think> I might have let errors slip in or failed to notice important points.",
#     5: "<think> Perhaps I misjudged something important or left out key information.",
#     6: "<think> On reflection, I may have skipped critical data or reasoned incorrectly.",
#     7: "<think> I could have missed significant insights or taken a wrong step.",
#     8: "<think> I sense a gap in the facts or a slip in my logic.",
#     9: "<think> I may have misinterpreted key facts or overlooked necessary details.",
#     10: "<think> My reasoning might be shaky or missing crucial context.",
#     11: "<think> I should revisit this—maybe I skipped vital info or made an error.",
#     12: "<think> There may be unnoticed gaps or mistakes in what I just reasoned.",
#     13: "<think> I suspect I overlooked or misunderstood something important.",
#     14: "<think> My earlier logic may hide errors or omit essential points.",
#     15: "<think> I may have missed key aspects or made subtle mistakes.",
#     16: "<think> I might have ignored essential details or misread part of it.",
#     17: "<think> On second look, oversights or errors may have slipped through.",
#     18: "<think> Important points might have escaped me, or I miscalculated.",
#     19: "<think> I likely missed something crucial or stumbled in my reasoning.",
#     20: "<think> Thinking back, I could have overlooked clues or judged incorrectly.",
#     21: "<think> Maybe I misunderstood a core idea or missed key evidence.",
#     22: "<think> I'm concerned about hidden errors or overlooked essentials.",
#     23: "<think> Perhaps that step wasn't fully accurate or lacked key details.",
#     24: "<think> It's conceivable I neglected critical information or made a mistake.",
#     25: "<think> I may have dismissed or misread something important.",
#     26: "<think> My previous reasoning might contain blind spots or quiet errors.",
#     27: "<think> I'm unsure whether I missed crucial points or made mistakes earlier.",
#     28: "<think> I feel uneasy—critical details may have slipped by or been misread.",
#     29: "<think> I recognize possible gaps or missteps I didn't catch before."
# }



action_prompts = {
    0: " Now I need to choose an action.",
    1: " I'll decide my next action carefully.",
    2: " I should clarify my next move.",
    3: " Now must select my next step.",
    4: " I'll choose what to do next.",
    5: " Action selection needed.",
    6: " I need to determine my next action.",
    7: " Prompting me to choose my next step.",
    8: " It's essential I select a clear action.",
    9: " I'll carefully choose my next action.",
    10: " I now need to explicitly decide my next action.",
    11: " I should pause and choose what action comes next.",
    12: " Determining my next step is essential.",
    13: " I'll now pick an action to address this.",
    14: " I should clarify and explicitly select my next action.",
    15: " I'll choose my next action.",
    16: " I now clearly state what action I'll take next.",
    17: " It's necessary to explicitly choose my next action.",
    18: " Explicitly pick the most appropriate next step.",
    19: " It's critical that I explicitly state my next action.",
    20: " It's important I now carefully choose my action.",
    21: " I must clearly choose and declare my next step.",
    22: " Now I'll pick an action explicitly.",
    23: " I'll explicitly choose my next action.",
    24: " I'll take a moment and explicitly choose the right action.",
    25: " I will now explicitly select my next step.",
    26: " It's necessary that I explicitly decide what to do next.",
    27: " Prompting me to explicitly state my next action.",
    28: " I'll pause and explicitly pick the best next action.",
    29: " It's important that I clearly identify my next step."
}