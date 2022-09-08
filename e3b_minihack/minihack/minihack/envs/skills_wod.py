# Copyright (c) Facebook, Inc. and its affiliates.
from minihack import MiniHackSkill, LevelGenerator, RewardManager
from minihack.envs import register
from nle import nethack


class MiniHackWoDEasy(MiniHackSkill):
    """Environment for "Wand of death" task."""

    def __init__(self, *args, **kwargs):
        map = """
|----------
|.........+
|----------
"""
        lvl_gen = LevelGenerator(map=map, lit=True)

        lvl_gen.set_start_pos((1, 1))
        kwargs["autopickup"] = kwargs.pop("autopickup", True)
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 50)

        lvl_gen.add_object(
            name="death", symbol="/", cursestate="blessed", place=((1, 1))
        )
        lvl_gen.add_monster("minotaur", args=("asleep",), place=(9, 1))

        des_file = lvl_gen.get_des()

        rwrd_mngr = RewardManager()
        rwrd_mngr.add_kill_event("minotaur")

        super().__init__(
            *args, des_file=des_file, reward_manager=rwrd_mngr, **kwargs
        )


class MiniHackWoDMedium(MiniHackSkill):
    def __init__(self, *args, **kwargs):
        map = """
|---------------------------|
|...........................|
|---------------------------|
"""
        lvl_gen = LevelGenerator(map=map, lit=True)

        lvl_gen.set_start_pos((1, 1))
        lvl_gen.add_goal_pos((27, 1))

        lvl_gen.add_object(
            name="death", symbol="/", cursestate="blessed", place=((2, 1))
        )
        lvl_gen.add_monster("minotaur", args=("asleep",), place=(26, 1))
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 150)

        des_file = lvl_gen.get_des()
        super().__init__(*args, des_file=des_file, **kwargs)


class MiniHackWoDMediumRestrictedActions(MiniHackSkill):
    def __init__(self, *args, **kwargs):
        map = """
|---------------------------|
|...........................|
|---------------------------|
"""
        lvl_gen = LevelGenerator(map=map, lit=True)

        lvl_gen.set_start_pos((1, 1))
        lvl_gen.add_goal_pos((27, 1))

        lvl_gen.add_object(
            name="death", symbol="/", cursestate="blessed", place=((2, 1))
        )
        lvl_gen.add_monster("minotaur", args=("asleep",), place=(26, 1))
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 150)

        ACTIONS = tuple(nethack.CompassDirection) + (nethack.Command.PICKUP, \
                                                     nethack.Command.ZAP, \
                                                     nethack.Command.FIRE)
        kwargs["actions"] = ACTIONS
        
        

        des_file = lvl_gen.get_des()
        super().__init__(*args, des_file=des_file, **kwargs)

        


class MiniHackWoDHard(MiniHackSkill):
    def __init__(self, *args, **kwargs):
        map = """
|---------------------------|
|...........................|
|.....|---------------------|
|.....|
|.....|
|-----|
"""
        lvl_gen = LevelGenerator(map=map, lit=True)

        lvl_gen.set_start_rect((1, 1), (5, 5))
        lvl_gen.add_goal_pos((27, 1))

        lvl_gen.set_area_variable("$safe_room", "fillrect", 1, 1, 5, 5)
        lvl_gen.add_object_area(
            "$safe_room", name="death", symbol="/", cursestate="blessed"
        )
        lvl_gen.add_monster("minotaur", place=(26, 1))
        des_file = lvl_gen.get_des()
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 400)

        super().__init__(*args, des_file=des_file, **kwargs)




class MiniHackWoDHardRestrictedActions(MiniHackSkill):
    def __init__(self, *args, **kwargs):
        map = """
|---------------------------|
|...........................|
|.....|---------------------|
|.....|
|.....|
|-----|
"""
        lvl_gen = LevelGenerator(map=map, lit=True)

        lvl_gen.set_start_rect((1, 1), (5, 5))
        lvl_gen.add_goal_pos((27, 1))

        lvl_gen.set_area_variable("$safe_room", "fillrect", 1, 1, 5, 5)
        lvl_gen.add_object_area(
            "$safe_room", name="death", symbol="/", cursestate="blessed"
        )
        lvl_gen.add_monster("minotaur", place=(26, 1))
        des_file = lvl_gen.get_des()
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 400)

        ACTIONS = tuple(nethack.CompassDirection) + (nethack.Command.PICKUP, \
                                                     nethack.Command.ZAP, \
                                                     nethack.Command.FIRE)
        kwargs["actions"] = ACTIONS
        

        super().__init__(*args, des_file=des_file, **kwargs)
        


class MiniHackWoDPro(MiniHackSkill):
    def __init__(self, *args, **kwargs):
        map = """
-------------------------------------
|.................|.|...............|
|.|-------------|.|.|.------------|.|
|.|.............|.|.|.............|.|
|.|.|----------.|.|.|------------.|.|
|.|.|...........|.|.............|.|.|
|.|.|.|----------.|-----------|.|.|.|
|.|.|.|...........|.......|...|.|.|.|
|.|.|.|.|----------------.|.|.|.|.|.|
|.|.|.|.|.................|.|.|.|.|.|
|.|.|.|.|.-----------------.|.|.|.|.|
|.|.|.|.|...................|.|.|.|.|
|.|.|.|.|--------------------.|.|.|.|
|.|.|.|.......................|.|.|.|
|.|.|.|-----------------------|.|.|.|
|.|.|...........................|.|.|
|.|.|---------------------------|.|.|
|.|...............................|.|
|.|-------------------------------|.|
|...................................|
-------------------------------------
"""
        lvl_gen = LevelGenerator(map=map, lit=True)
        lvl_gen.set_start_pos((19, 1))
        lvl_gen.add_goal_pos((19, 7))
        lvl_gen.add_monster(name="minotaur", place=(19, 9))
        lvl_gen.add_object("death", "/", cursestate="blessed")
        des_file = lvl_gen.get_des()
        kwargs["max_episode_steps"] = kwargs.pop("max_episode_steps", 1000)

        super().__init__(
            *args,
            des_file=des_file,
            **kwargs,
        )


register(
    id="MiniHack-WoD-Easy-v0",
    entry_point="minihack.envs.skills_wod:MiniHackWoDEasy",
)

register(
    id="MiniHack-WoD-Medium-v0",
    entry_point="minihack.envs.skills_wod:MiniHackWoDMedium",
)

register(
    id="MiniHack-WoD-Medium-v1",
    entry_point="minihack.envs.skills_wod:MiniHackWoDMediumRestrictedActions",
)

register(
    id="MiniHack-WoD-Hard-v0",
    entry_point="minihack.envs.skills_wod:MiniHackWoDHard",
)


register(
    id="MiniHack-WoD-Hard-v1",
    entry_point="minihack.envs.skills_wod:MiniHackWoDHardRestrictedActions",
)


register(
    id="MiniHack-WoD-Pro-v0",
    entry_point="minihack.envs.skills_wod:MiniHackWoDPro",
)
