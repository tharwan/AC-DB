import warnings
import numpy as np
import pickle
from recordclass import recordclass
import logging
from pandas import DataFrame, Series
import networkx as nx

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
np.set_printoptions(precision=3)

from scipy.signal import gaussian

import matplotlib.pyplot as plt
from cython_utils import calcDemandCython, randomWalk, calcProfiles
from recorder import Recorder


LENGTH_OF_DAY = 24 * 60

ERROR_SCALE = 0.01
_MATCHER = "METRO"
_PROD_SURPLUS = True
_LOG_MATCHER = False
_ANNEAL_TEMP = 10000
_MATCHER_CMP_ALL = False

MAX_DEMAND_DIFF = 1  # MWh in 15 minutes

OVERCAPACITY_FACTOR = 1.1
USER_OPTIMIZE_PROB = 0.95
DEMAND_HISTORY_LEN = 30

CONSUMER_SPOT_DECAY = 1  # 0 - 1, where 0 means consider all 30 days equal,
# 1 means consider only the latest entry
UTILITY_DEMAND_DECAY = 0.5

OPTIMIZERS = 0.2  # percentage of optimizing consumer agents
REMOTE_CONTROLLED = 1.0  # percentage of optimizing users that the utility controlls

CAPACITY_LIMIT = 500

SOLAR_ERROR = 0.1
WIND_ERROR = 0.1
RENEWABLE_SAFETY_MARGIN = 1
RENEWABLE_REG_FACTOR = 0
RENEWABLE_ERROR_1 = 0.03
RENEWABLE_ERROR_2 = 0.03
RENEWABLE_CORRELATION = 1.0

SPOT_MARKET_RESOLUTION = 60
BALANCING_INTERVALL = 15
BALANCING_MIN_OFFER = 0.5


REGULATION_FACTOR = 0.5
REGULATION_PRICE_FACTOR = 0.2
MIN_RUN_FACTOR = 0
MAX_SPOT_BID = 0.95

USE_CACHE_AGENTS = False


LOG_DEMAND = False

_MPI = False
# TODO: these need to be updated automatically
_SLICES = int(LENGTH_OF_DAY / SPOT_MARKET_RESOLUTION)
_B_SLICES = int(SPOT_MARKET_RESOLUTION / BALANCING_INTERVALL)
# A word about units: powerplants are all full MW. demand is in kW.
# Since the market deals only in full MW the utility also deals in full MW.
# Therefore the utility does most of the conversion

# a slice is the part of the day given by SPOT_MARKET_RESOLUTION
# a subslice is the part of a slice given by BALANCING_INTERVALL


Offer = recordclass('Offer', 'max min price idx t node')
Profile = recordclass('Profile', 'profile price idx id base')
ScheduleItem = recordclass('ScheduleItem', 'accepted min max price idx t')
AgentCache = recordclass('AgentCache', 'normal optimizing')
RegulationOffer = recordclass(
    "RegulationOffer", "up down priceUp priceDown idx t")
Correction = recordclass("Correction", "capacity power idx t")


def slice2T(slc, blcSlc=None, full=False):
    if blcSlc is None:
        t_start = slc * SPOT_MARKET_RESOLUTION
        return np.arange(t_start, t_start+SPOT_MARKET_RESOLUTION)
    else:
        if full:
            t_start = slc * SPOT_MARKET_RESOLUTION + blcSlc * BALANCING_INTERVALL
            return np.arange(t_start, t_start+BALANCING_INTERVALL * (_B_SLICES - blcSlc))
        else:
            t_start = slc * SPOT_MARKET_RESOLUTION + blcSlc * BALANCING_INTERVALL
            return np.arange(t_start, t_start+BALANCING_INTERVALL)


def HtoMin(H):
    t_len = LENGTH_OF_DAY/24
    t_start = H * t_len
    t_end = t_start + t_len
    t = np.arange(t_start, t_end, dtype=int)
    return t


def HqToMin(H, q):
    if q not in [0, 1, 2, 3]:
        raise ValueError("the quarter must be: 0 <= q <= 3.")
    t = HtoMin(H).reshape(4, 15)
    return t[q]


def integrateDay15min(values):
    return np.sum(values.reshape(96, 15), axis=1)


def integrateDayH(values):
    return np.sum(values.reshape(24, 60), axis=1)


def integrateDayQtoH(values):
    return np.sum(values.reshape(24, 4), axis=1)


def integrateDayToSlice(values):
    return np.sum(values.reshape(_SLICES, SPOT_MARKET_RESOLUTION), axis=1)


def integrateDayToSubslice(values):
    return np.sum(values.reshape(_SLICES*_B_SLICES, BALANCING_INTERVALL), axis=1)


class System(Recorder):
    """the class that controls the whole system, timing loop and everything,
    """

    def __init__(self):
        super().__init__()

        self.users = []

        self.utilities = []

        self.producers = []

        self.spot = MarketAgent(self)

        self._dailyDemand = None
        self._dailyForecastedDemand = None
        self._minuteDailyDemand = None

        self._minutesProdSchedule = None

        self._t0_history = []
        self._demandHistory = []
        self._diffHistory = []
        self._spotPriceHistory = []
        self._spotPriceAverageCache = None
        self._balancingPriceHistory = []
        self._regulationPriceHistory = []
        self._regulationTypeHistory = []

        self._renewableCorrelation = randomWalk(
            LENGTH_OF_DAY, 1, RENEWABLE_ERROR_1, RENEWABLE_ERROR_2)
        self.solarTime = np.random.randint(3*60)

        self.zones_dict = {0:0}
        self.network = nx.Graph()
        self.network.add_node(0)


    def node2zone(self,node):
        if node in zones_dict:
            return self.zones_dict[node]
        else:
            raise ValueError("No zone defined for node {}".format(node))

    def cycle_matrix(self):
        cycles = nx.cycle_basis(self.network)
        edges = [e for e in self.network.edges]
        nodes = [n for n in self.network.nodes]

        cycle_matrix = np.zeros((len(edges),len(cycles)))

        for c_idx,c in enumerate(cycles):
            c_edges = cycle2edgeList(c)
            for e_idx,e in enumerate(edges):
                if e in c_edges:
                    cycle_matrix[e_idx,c_idx] = 1
                if e[::-1] in c_edges:
                    cycle_matrix[e_idx,c_idx] = -1

        return cycle_matrix

    def incidence_matrix(self,nodes,edges):
        i_mat = nx.incidence_matrix(
                            self.network,
                            nodelist=nodes,
                            edgelist=edges,
                            oriented=True)
        return -i_mat.toarray()


    @property
    def spotPrice(self):
        return self._spotPriceHistory[-1]

    @spotPrice.setter
    def spotPrice(self, value):
        # todo: this should not have a setter
        self._spotPriceAverageCache = None
        self._spotPriceHistory.append(value)

    def spotPriceHAverage(self, c=0, l=30):
        if len(self._spotPriceHistory) > 1:
            if self._spotPriceAverageCache is None:
                self._spotPriceAverageCache = dict()

            if c in self._spotPriceAverageCache:
                return self._spotPriceAverageCache[c]

            else:
                x = np.array(self._spotPriceHistory[-l:])
                w = (np.ones(x.shape)-c) ** (np.repeat(
                    np.arange(x.shape[0]-1, -1, -1), x.shape[1]
                )
                    .reshape(x.shape))
                w /= np.sum(w, axis=0)
                self._spotPriceAverageCache[c] = np.sum(x*w, axis=0)
                return self._spotPriceAverageCache[c]
        else:
            return self._spotPriceHistory[0]

    def nodalPriceForBus(self, bus):
        if hasattr(self, 'nodalPrices'):
            return self.nodalPrices[:, bus]
        else:
            raise ValueError("Nodal prices not found! Check setup.")

    @property
    def daysRun(self):
        return len(self._spotPriceHistory)

    @property
    def dailyForecastedDemand(self):
        if self._dailyForecastedDemand is None:
            demand = np.zeros(_SLICES)
            for util in self.utilities:
                D = util.forecastDailyDemand()
                demand += D

                logger.debug("util {} forecasted:{}".format(util.idx, str(D)))
            self._dailyForecastedDemand = demand
        return self._dailyForecastedDemand

    @property
    def dailyDemand(self):
        if self._dailyDemand is None:
            demand = np.zeros(_SLICES)
            for util in self.utilities:
                demand += util.dailyDemand
            self._dailyDemand = demand
        return self._dailyDemand

    @property
    def minuteDailyDemand(self):
        if self._minuteDailyDemand is None:
            demand = np.zeros(LENGTH_OF_DAY)
            t = np.arange(LENGTH_OF_DAY)
            for util in self.utilities:
                demand += sum(util.minuteDemandSplit(current=True))
            self._minuteDailyDemand = demand
        return self._minuteDailyDemand

    def plotPriceMW(self):
        H = 0
        supply = [[p.priceForH(H), p.upCapacityForH(H)]
                  for p in self.producers]
        supply.sort()
        price, cap = zip(*supply)
        sum_cap = np.cumsum(cap)

        plt.figure()
        plt.plot(sum_cap, price)
        plt.show()

    def plotPriceDay(self):
        if not(self.spotPrice is None):
            plt.figure()
            plt.plot(self.spotPrice)
        else:
            logger.warning("No price for the day, yet.")

    def runSpotMarket(self):
        prices = np.zeros(_SLICES)
        producerSchedule = {}

        if _MATCHER == "METRO":
            selectedProfiles, profile, accepted, prices = self.spot.matchDemandForDayR()
        elif _MATCHER == "MIP":
            selectedProfiles, profile, accepted, prices = self.spot.matchDemandForDayMIP()
        else:
            raise ValueError("No _MATCHER defined")

        for slc_sItems in accepted:
            for sItem in slc_sItems:

                if sItem.idx in producerSchedule:
                    producerSchedule[sItem.idx][sItem.t] = sItem.accepted
                else:
                    producerSchedule[sItem.idx] = np.zeros(LENGTH_OF_DAY)
                    producerSchedule[sItem.idx][sItem.t] = sItem.accepted

        return prices, producerSchedule, selectedProfiles

    def calculateBalancingCosts(self, balacingPrices, balancingType, allCorrections):

        producerRev = np.zeros(_SLICES*_B_SLICES)

        # producer Rev due to corrections
        for num, corrections in enumerate(allCorrections):
            slc = num // _B_SLICES
            blcSlc = num % _B_SLICES

            for c in corrections:
                diff = c.power
                if balancingType[slc] == 'd':
                    if diff <= 0:  # actual down regulation
                        price = balacingPrices[slc][1]
                    elif diff > 0:  # up regulation in down hour
                        price = self.spotPrice["sys"][slc]
                elif balancingType[slc] == 'u':
                    if diff >= 0:  # actual up regulation
                        price = balacingPrices[slc][0]
                    elif diff < 0:  # down regulation in up hour
                        price = self.spotPrice["sys"][slc]
                else:
                    price = self.spotPrice["sys"][slc]

                self.producers[c.idx].balancingRev += price * diff
                producerRev[num] += price * diff

        logger.debug(
            "producer revenues for every quarter:\n{}".format(str(producerRev)))

        # util cost
        allDiff = np.zeros(_SLICES*_B_SLICES)
        utilRev = np.zeros(_SLICES*_B_SLICES)
        for util in self.utilities:
            forecast = np.repeat(util.forecastDailyDemand(
            )/SPOT_MARKET_RESOLUTION, SPOT_MARKET_RESOLUTION)
            demand = sum(util.minuteDemandSplit(
                current=True)) / 60  # MWm to MWh

            logger.debug("balancing for {}".format(util.idx))

            diff = integrateDayToSubslice(demand - forecast)
            allDiff += diff

            cost = 0
            for num, d in enumerate(diff):
                slc = num // _B_SLICES
                blcSlc = num % _B_SLICES

                if balancingType[slc] == 'd':
                    price = balacingPrices[slc][1]
                elif balancingType[slc] == 'u':
                    price = balacingPrices[slc][0]
                else:
                    price = self.spotPrice["sys"][slc]

                cost += price * d
                utilRev[num] += price * d
                logger.debug(
                    "\tin {}/{}: {:7.3f}MWh * {:7.3f}€/MWh = {:7.3f}€".format(slc, blcSlc, d, price, price*d))

            util.balancingCost += cost

            # print("balancingCost {}".format(cost))
            logger.debug("sum of diff:{}".format(str(allDiff)))
            logger.debug(
                "util cost for every quarter:\n{}".format(str(utilRev)))
            logger.debug("sum of util cost:{}".format(sum(utilRev)))

        # producer cost
        for p in self.producers:
            schedule = p.minuteSchedule()
            production = p.minuteProduction()

            diff = integrateDayToSubslice(
                production - schedule)/60  # MWm to MWh
            rev = 0

            for num, d in enumerate(diff):
                slc = num // _B_SLICES
                blcSlc = num % _B_SLICES

                if balancingType[slc] == 'd':
                    if d <= 0:  # under produciton in down hour
                        price = self.spotPrice["sys"][slc]
                    elif d > 0:  # over production in down hour
                        price = balacingPrices[slc][1]
                elif balancingType[slc] == 'u':
                    if d >= 0:  # over produciton in up hour
                        price = self.spotPrice["sys"][slc]
                    elif d < 0:  # under produciton in up hour
                        price = balacingPrices[slc][0]
                else:
                    price = self.spotPrice["sys"][slc]

                rev += price * d

            p.balancingRev += rev

    def calculateUtilityCost(self, hourlyPrices):

        for util in self.utilities:
            demand = util.forecastDailyDemand()
            demand = np.round(demand, 4)  # we round to the 4 decimal ...
            # to be consitend with users paying their kWh to the first decimal

            logger.debug("demand of utility {} is {}".format(
                util.idx, str(demand)))
            logger.debug("cost per H:{}".format(str(demand*hourlyPrices[util.node])))
            logger.debug("sum: {}".format(
                np.sum(np.round(demand*hourlyPrices[util.node], 2))))
            util.purchaseCost += np.sum(np.round(demand*hourlyPrices[util.node], 2))

    def calculateProducerRevenues(self, hourlyPrices, producerSchedule):

        for p in self.producers:
            if p.idx in producerSchedule:
                # we sum up for every minute so we devide by the number
                # of minutes per slice again
                schedule = integrateDayToSlice(
                    producerSchedule[p.idx]/(LENGTH_OF_DAY/_SLICES))
                p.revenues += np.sum(schedule*hourlyPrices[p.node])

    def performBalancing(self):
        day_diff = self.forecastDemandDiff()
        # diff in subslices
        day_diffBlc = np.round(
            integrateDayToSubslice(day_diff)/60)  # MWm to MWh
        # diff in slices
        day_diffSlice = np.round(
            integrateDayToSlice(day_diff)/60)  # MWm to MWh

        max_diff = MAX_DEMAND_DIFF / 15 * BALANCING_INTERVALL

        logger.debug(
            "balancing needed per subslice:{}".format(str(day_diffBlc)))

        demand = self.minuteDailyDemand

        balancingType = []
        pricePerH = []
        allCorrections = [[]
                          for i in range(int(LENGTH_OF_DAY/BALANCING_INTERVALL))]
        for slc in range(_SLICES):
            t_slc = slice2T(slc)

            # offers = [p.regulationOfferForT(t_slc) for p in self.producers]
            # offers = list(filter(lambda x: x, offers))  # removes all None

            priceUp = self.spotPrice["sys"][slc]
            priceDown = self.spotPrice["sys"][slc]
            matched = 0

            for blcSlc in range(_B_SLICES):
                diff = day_diffBlc[_B_SLICES*slc+blcSlc]
                accepted = []
                t_blc = slice2T(slc, blcSlc, full=True)

                offers = [p.regulationOfferForT(t_blc) for p in self.producers]
                offers = list(filter(lambda x: x, offers))  # removes all None

               

                if diff - matched > max_diff:
                    # up regulation needed
                    # print "up regulation {} in {}/{}".format(diff,H,q)

                    while matched < diff:

                        if len(offers) == 0:
                            
                            raise ValueError(
                                "system could not be up balanced! {} MWh diff {} MWh matched".format(diff,matched))

                        offers.sort(key=lambda x: x.priceUp)

                        # find lowest up offer with value bigger 0
                        idx = 0
                        while True:
                            if offers[idx].up > 0:
                                offer = offers.pop(idx)
                                break
                            else:
                                idx += 1
                                if idx >= len(offers):
                                    logger.error(offers)
                                    raise ValueError(
                                        "system could not be up balanced! {} missing".format(diff-matched))

                        priceUp = max((offer.priceUp, priceUp))

                        power = offer.up * BALANCING_INTERVALL / 60  # MWm to MWh

                        if power > diff-matched:
                            accepted_capacity = \
                                (diff-matched) / BALANCING_INTERVALL * 60

                            matched += diff-matched

                        else:

                            matched += power
                            accepted_capacity = offer.up

                        accepted_power = accepted_capacity * \
                            (t_blc[-1] - t_blc[0])
                        accepted.append(Correction(accepted_capacity,
                                                   accepted_power,
                                                   offer.idx,
                                                   t_blc))

                elif diff - matched < -max_diff:
                    # down regulation needed
                    # print "down regulation {} in {}/{}".format(diff,H,q)

                    while matched > diff:

                        offers.sort(key=lambda x: x.priceDown, reverse=True)

                        # find highest down offer with value bigger 0
                        # not that the sort was reversed

                        idx = 0
                        while True:

                            if len(offers) == 0 or idx >= len(offers):
                                logger.warn(
                                    "system could not be down balanced!")
                                logger.warn("price set to 0.")

                                # TODO: negative prices and such
                                matched = diff
                                priceDown = 0
                                break  # no more elements in the list
                            try:
                                if offers[idx].down > 0:  # if it is a down offer
                                    # unpack the offer
                                    offer = offers.pop(idx)

                                    priceDown = min(
                                        (offer.priceDown, priceDown))
                                    # NOTE: it is possible for the offer price to be higher
                                    # then the current down price, since the offer can be
                                    # from a plant that is just in the list because if got
                                    # activated for down regulation

                                    power = -offer.down * BALANCING_INTERVALL / 60  # MWm to MWh

                                    if power <= diff-matched:
                                        accepted_capacity = \
                                            (diff-matched) / \
                                            BALANCING_INTERVALL * 60

                                        matched += diff-matched
                                        break  # this offer did match our needs fully we can stop looking
                                    else:

                                        matched += power
                                        accepted_capacity = -offer.down

                                    # append offer to the accepted ones
                                    accepted_power = -accepted_capacity\
                                        * (t_blc[-1] - t_blc[0])
                                    accepted.append(Correction(accepted_capacity,
                                                               accepted_power,
                                                               offer.idx,
                                                               t_blc))
                                else:  # no down offer so advance
                                    idx += 1

                            except IndexError:
                                logger.error(len(offers), idx)
                                raise

                for c in accepted:
                    self.producers[c.idx].applyBlcCorrectionForT(
                        c.t, c.capacity)

                allCorrections[_B_SLICES*slc+blcSlc] = accepted

            for p in self.producers:
                p.nextSlc()

            pricePerH.append([priceUp, priceDown])

            if day_diffSlice[slc] > 1:
                balancingType.append('u')
            elif day_diffSlice[slc] < -1:
                balancingType.append('d')
            else:
                balancingType.append('n')

        return pricePerH, balancingType, allCorrections

    def setProducerSchedule(self, schedule):
        for idx, caps in schedule.items():
                # self.producers[idx].setSchedule(caps)
            self.producers[idx].setMinuteSchedule(caps)

    def setUtilityProfiles(self, profiles):

        for idx, profile in profiles.items():
            assert idx == self.utilities[idx].idx
            # if this does not hold true allways we need to adress it by having
            # a dict about the utilities and their indexes
            self.utilities[idx].setProfileForDay(profile)

    def minuteDailyProduction(self):
        if self._minutesProdSchedule is None:
            if np.all(np.isnan(self.spotPrice[0])):
                raise ValueError("can not get daily production yet.")
            schedule = np.zeros(LENGTH_OF_DAY)
            for p in self.producers:
                schedule += p.minuteProduction()
            self._minutesProdSchedule = schedule
        return self._minutesProdSchedule

    def updateProductionScheduleForT(self, t, value):
        # print('update production schedule: {} for {}'.format(value,t))
        self._minutesProdSchedule[t] += value

    def forecastDemandDiff(self):

        return self.minuteDailyDemand - self.minuteDailyProduction()

    def runForDays(self, days):
        for d in range(days):

            logger.info("day {}".format(d))
            self.startDay()

            # ==== spot market =====
            hourlyPrices, producerSchedule, selectedProfiles = self.runSpotMarket()

            # print(len(producerSchedule))
            # for idx,item in producerSchedule.items():
            #   print(idx,item)

            self.setProducerSchedule(producerSchedule)
            self.setUtilityProfiles(selectedProfiles)

            # for util in self.utilities:
            #   print(util._profile)

            self.spotPrice = hourlyPrices


            self.calculateUtilityCost(hourlyPrices)
            self.calculateProducerRevenues(hourlyPrices, producerSchedule)

            for util in self.utilities:
                util.updatePostMarket()

            self._diffHistory.append(self.forecastDemandDiff())
            self._minuteDailyDemand = None  # this might have changed due to the update

            

            # ====== balancing ========
            balacingPrices, balancingType, allCorrections = self.performBalancing()
            self.calculateBalancingCosts(
                balacingPrices, balancingType, allCorrections)

            # ^===== balancing =======
            # plt.plot(self.minuteDailyProduction())
            # self.plotPriceDay()

            bprice = []
            for t, p, sp in zip(balancingType, balacingPrices, self.spotPrice):
                if t == 'u':
                    bprice.append(p[0])
                elif t == 'd':
                    bprice.append(p[1])
                else:
                    bprice.append(sp)

            self._balancingPriceHistory.append(list(zip(*balacingPrices)))
            self._regulationPriceHistory.append(bprice)
            self._regulationTypeHistory.append(balancingType)

            t0 = np.fromiter((u.t0 for u in self.users),
                             float, len(self.users))
            self._t0_history.append(t0)

            self.nextDay()
            logger.info('...done')

    def startDay(self):
        self._renewableCorrelation = randomWalk(
            LENGTH_OF_DAY, 1, RENEWABLE_ERROR_1, RENEWABLE_ERROR_2)
        self.solarTime = np.random.randint(3*60)
        for util in self.utilities:
            util.startDay()

    def nextDay(self):

        if self.daysRun > 0:
            self._demandHistory.append(self.minuteDailyDemand)

        for util in self.utilities:
            util.nextDay()

        for p in self.producers:
            p.nextDay()

        self._dailyDemand = None
        self._dailyForecastedDemand = None
        self._minuteDailyDemand = None
        self._minutesProdSchedule = None

    def resetCosts(self):

        for util in self.utilities:
            util.resetCosts()

        for p in self.producers:
            p.resetCosts()

        for u in self.users:
            u.resetCosts()

    def costPerMW(self):
        userCostV = []
        optiUserCostV = []
        userCost = []
        optiUserCost = []

        fcDict = {util.idx: util.fixedCosts() for util in self.utilities}
        usageDict = {}

        for util in self.utilities:
            optUsage = 0
            usage = 0

            for u in util.users:

                if u.isOptimizer:
                    optUsage += u.usedPower/1000
                else:
                    usage += u.usedPower/1000

            usageDict[util] = (usage, optUsage)

            cost = util.costs()
            userCostV.append(cost['normal_variable'])
            optiUserCostV.append(cost['opt_variable'])
            fcMW = fcDict[util.idx] / (usage+optUsage)
            if usage > 0:
                userCost.append((cost['normal_variable'] + fcMW * usage))
            if optUsage > 0:
                optiUserCost.append((cost['opt_variable'] + fcMW * optUsage))

        ret = dict()
        ret["usageNormal"] = sum(item[0] for key, item in usageDict.items())
        ret["usageOpt"] = sum(item[1] for key, item in usageDict.items())
        ret["costNormalVari"] = np.sum(userCostV)
        ret["costNormalAll"] = np.sum(userCost)
        ret["costOptVari"] = np.sum(optiUserCostV)
        ret["costOptAll"] = np.sum(optiUserCost)
        ret["costMW"] = (ret["costOptAll"] + ret["costNormalAll"]
                         )/(ret["usageNormal"]+ret["usageOpt"])

        return ret

    def saveScenario(self, name):

        # users
        t0 = []
        maxD = []
        minD = []
        isOptimizer = []
        util = []
        for u in self.users:
            t0.append(u.t0)
            maxD.append(u.maxD)
            minD.append(u.minD)
            isOptimizer.append(u._isOptimizer)
            util.append(u.util_idx)

        # utilities
        # nothin?

        # producers
        pidx = []
        capacity = []
        price = []
        cost = []
        Ptype = []
        minRunFactor = []
        regulationFactor = []
        for p in self.producers:
            pidx.append(p.idx)
            capacity.append(p.maxCapacity)
            price.append(p.priceMW)
            cost.append(p.cost)
            minRunFactor.append(p.minRunFactor)
            regulationFactor.append(p.regulationFactor)
            if isinstance(p, SolarAgent):
                Ptype.append('solar')
            elif isinstance(p, WindAgent):
                Ptype.append('wind')
            else:
                Ptype.append('normal')

        out = {
            't0': np.array(t0),
            'maxD': np.array(maxD),
            'minD': np.array(minD),
            'isOptimizer': np.array(isOptimizer),
            'util': np.array(util),
            'pidx': np.array(pidx),
            'capacity': np.array(capacity),
            'price': np.array(price),
            'cost': np.array(cost),
            'Ptype': np.array(Ptype),
            'minRunFactor': np.array(minRunFactor),
            'regulationFactor': np.array(regulationFactor)
        }

        np.savez(name, **out)


    @classmethod
    def loadScenario(cls, name):
        sys = cls()
        scenario = np.load(name)

        t0 = scenario['t0']
        maxD = scenario['maxD']
        minD = scenario['minD']
        isOptimizer = scenario['isOptimizer']
        util = scenario['util']

        userUtilDict = dict()

        # create users
        for t, maD, miD, opt, util in zip(t0, maxD, minD, isOptimizer, util):
            user = DemandAgent(sys, maD, miD, t, opt)
            sys.users.append(user)

            if util in userUtilDict:
                userUtilDict[util].append(user)
            else:
                userUtilDict[util] = [user]

        # create utilities
        for idx, users in userUtilDict.items():
            sys.utilities.append(UtilityAgent(sys, users, idx))

        # create producers

        pidx = scenario['pidx']
        capacity = scenario['capacity']
        price = scenario['price']
        cost = scenario['cost']  # ignored further on
        Ptype = scenario['Ptype']

        if 'minRunFactor' in scenario:
            minRun = scenario['minRunFactor']
        else:
            minRun = np.ones(cost.shape)*MIN_RUN_FACTOR
        if 'regulationFactor' in scenario:
            regulationFactor = scenario['regulationFactor']
        else:
            regulationFactor = np.ones(cost.shape)*REGULATION_FACTOR
        number = np.arange(len(cost))

        for idx, cap, p, c, typ, i in zip(pidx, capacity, price, cost, Ptype, number):
            if typ == 'solar':
                sys.producers.append(SolarAgent(sys, idx, np.max(cap), p))
            elif typ == 'wind':
                sys.producers.append(WindAgent(sys, idx, np.max(cap), p))
            else:
                sys.producers.append(ProductionAgent(sys, idx, np.max(cap), p))

            sys.producers[i].minRunFactor = minRun[i]
            sys.producers[i].regulationFactor = regulationFactor[i]

        return sys

    def saveData(self, name):
        out = dict()
        out['phase'] = np.array(self._t0_history)
        out['balancingPower'] = np.array(self._diffHistory)
        out['spotPrice'] = np.array(self._spotPriceHistory)

        if hasattr(self, "capacity"):
            out['capacity'] = self.capacity
            out['renewables'] = self.renewable_capacity
            out['needed_capacity'] = self.needed_capacity
        else:
            out['capacity'] =\
                sum([p.maxCapacity for p in self.producers if not(p.isRenewable())])
            out['renewables'] =\
                sum([p.maxCapacity for p in self.producers if p.isRenewable()])
            out['needed_capacity'] =\
                sum((u.maxD for u in self.users))/1000.0

        userCost = []
        optiUserCost = []
        variUserCost = []
        variOptiUserCost = []
        fcDict = {util.idx: util.fixedCostPerUser() for util in self.utilities}
        for u in self.users:
            if u._isOptimizer:
                optiUserCost.append((u.cost+fcDict[u.util_idx])/self.daysRun)
                variOptiUserCost.append((u.cost)/self.daysRun)
            else:
                userCost.append((u.cost+fcDict[u.util_idx])/self.daysRun)
                variUserCost.append((u.cost)/self.daysRun)

        out['normalUserCost'] = np.array(userCost)
        out['optiUserCost'] = np.array(optiUserCost)
        out['variOptiUserCost'] = np.array(variOptiUserCost)
        out['variUserCost'] = np.array(variUserCost)

        priceMW = []
        for p in self.producers:
            if p.isRenewable():
                priceMW.append(p.priceMW_all)
        out['priceMW'] = np.array(priceMW)

        # save all uppercase variables in a dict
        out['config'] = {k: v for k, v in globals().items()
                         if k.isupper() and k[0] != '_'}

        with open(name, 'wb') as outfile:
            pickle.dump(out, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    def renewableError(self):
        ret = randomWalk(LENGTH_OF_DAY, 1, RENEWABLE_ERROR_1,
                         RENEWABLE_ERROR_2)
        ret += RENEWABLE_CORRELATION * self._renewableCorrelation
        ret /= 1 + RENEWABLE_CORRELATION
        return ret


class MarketAgent(object):
    # TODO: clean up no longer needed functionality, like matchDemandFor...

    def __init__(self, system):
        self.system = system
        self.selection = None

    def pricesForProfile(self, profile, offers):
        prices = np.zeros_like(profile)
        prod_surplus = 0
        for slc, usage in enumerate(profile):
            price, accepted = self.priceForUsage(usage, offers[slc])
            for item in accepted:
                prod_surplus += item.accepted * (price - item.price)
            prices[slc] = price
        return prices, prod_surplus

    def priceForUsage(self, usage, sliceOffers, offers=False):
        available = 0
        accepted = []
        for o in sliceOffers:
            power = o.max * SPOT_MARKET_RESOLUTION / 60  # from MWm to MWh
            # if offers we keep track of the accepted offers from producers
            if offers:
                acc_p = power
                if power > usage-available:
                    acc_p = usage-available
                acc = acc_p / SPOT_MARKET_RESOLUTION * 60  # to MW
                item = ScheduleItem(acc, o.min, o.max,
                                    o.price, o.idx, o.t)
                accepted.append(item)
            available += power
            price = o.price

            if available >= usage:
                break
        return price, accepted

    def evaluateProfile(self, base, profile, n, offers):

        surplus = 0
        prices = []

        for slc, usage in enumerate(base):
            # iterate every hour and add the base load to the test profile
            # in fast mode we might scale up with n
            u = usage + profile[slc] * n
            price, acc = self.priceForUsage(u, offers[slc])
            prices.append(np.round(price))
            # TODO: this needs to be redone to consider different prices in the
            # different profiles
            surplus += (usage + profile[slc] * n) * (3000 - price)

        return np.round(surplus), prices



    def primeNetwork(self, country):
        network = self.system.network
        p_set = dict()
        power = country.generators * self.power_matrix
        for i in range(country.genSize):
            p_set["gen {}".format(i+1)] = power[:, i]
        network.generators_t.p_set = DataFrame(p_set)

        load_p_set = dict()
        for i in range(len(network.buses)):
            load_p_set["load {}".format(i)] = np.zeros(_SLICES)

        for u in self.system.users:
            profile = self.profiles[u.util_idx][country[u.util_idx]]
            util = self.system.utilities[u.util_idx]
            maxLoad = util.maxLoad["maxNorm"] + util.maxLoad["maxFlex"]
            load_p_set["load {}".format(
                int(u.bus))] += profile.profile * u.maxD / maxLoad

        network.loads_t.p_set = DataFrame(load_p_set)



    def suggest(self, country):
        self.primeNetwork(country)
        network = self.system.network
        cost = dict()
        # TODO: work in available power
        for i in range(country.genSize):
            cost["gen {}".format(i+1)] = self.price_matrix[0, i]
        network.generators.marginal_cost = Series(cost)
        network.lopf()
        gen_power = np.array(network.generators_t.p)
        return gen_power/self.power_matrix



    def matchDemandForDayR(self, n=5000):
        
        supply = []
        for slc in range(_SLICES):
            t = slice2T(slc)
            slc_supply = [p.spotOfferForT(t) for p in self.system.producers]
            slc_supply.sort(key=lambda x: x.price)
            supply.append(slc_supply)

        profiles = dict()
        nbr_profiles = np.zeros(len(self.system.utilities))
        for util in self.system.utilities:
            profiles[util.idx] = util.profilesForDay()
            nbr_profiles[util.idx] = len(profiles[util.idx])

        def profileForSelection(selection):
            p = np.zeros(_SLICES)
            for (idx,), s in np.ndenumerate(selection):
                prof = profiles[idx]
                p += prof[s].profile
                #p += profiles[idx][s].profile
            return p

        def eval(selection):
            profile = profileForSelection(selection)
            prices, prod_surplus = self.pricesForProfile(profile, supply)
            if _PROD_SURPLUS:
                return np.sum(profile * (3000 - prices)) + prod_surplus
            else:
                return np.sum(profile * (3000 - prices))

        if np.any(nbr_profiles > 0):

            if self.selection is None:
                selection = np.random.randint(
                    0, _SLICES+1, len(self.system.utilities))
                
            else:
                selection = self.selection
            selection = np.ones(len(self.system.utilities))*12
            selection -= np.random.randint(0,2,len(self.system.utilities))
            
            selection = np.mod(selection, nbr_profiles)
            value = eval(selection)
            logger.debug("initial selection value:{:,.0f}".format(value))
            best = (value.copy(), selection.copy())

            T = _ANNEAL_TEMP

            detour = 0
            for i in range(n):
                Tt = max((T/(i+1),1))
                length = np.ceil(np.random.exponential(3))
                move_vec = np.random.randint(-length, length+1, len(selection))
                new_selection = np.mod(selection + move_vec, nbr_profiles)
                new_value = eval(new_selection)
                if np.exp((new_value - value)/Tt) > np.random.rand() or new_value > value:
                    selection = new_selection

                    if new_value > best[0]:
                        logger.debug("best {} with {:,.0f} (diff:{:.2f} Tt:{:.0f})".format(
                            new_selection, new_value, new_value-best[0], Tt))
                        best = (new_value.copy(), new_selection.copy())

                    if new_value < value:
                        detour += 1

                    value = new_value

            logger.debug("detours:{}".format(detour))

            selection = best[1]

            logger.debug("selected value {:,.0f}".format(eval(selection)))

        else:
            selection = np.zeros(len(self.system.utilities))

        selectedProfiles = {}
        for (idx,), s in np.ndenumerate(selection):
            selectedProfiles[idx] = profiles[idx][s]
        profile = profileForSelection(selection)

        price = []
        accepted = []
        for slc, usage in enumerate(profile):
            p, acc = self.priceForUsage(usage, supply[slc], offers=True)
            price.append(p)
            accepted.append(acc)
        
        logger.info("price max:{:.2f} min:{:.2f} avg:{:.2f}".format(
            np.max(price), np.min(price), np.mean(price)))
        
        logger.info("overall cost:{:,.0f}".format(
            np.sum(np.array(price)*profile)))
        
        self.selection = selection

        return selectedProfiles, profile, accepted, price
    
    def matchDemandForDayMIP(self):
        import pulp as pp

        # network stuff
        cycles = nx.cycle_basis(self.system.network)
        edges = [e for e in self.system.network.edges]
        nodes = [n for n in self.system.network.nodes]
        cycle_matrix = self.system.cycle_matrix()
        incidence_matrix = self.system.incidence_matrix(nodes,edges)

        supply = []
        offer_vars = {}
        #collect all offers from producers
        for slc in range(_SLICES):
            t = slice2T(slc)
            slc_supply = [p.spotOfferForT(t) for p in self.system.producers]
            slc_supply = list(filter(lambda o: o.max > 0, slc_supply))
            slc_supply.sort(key=lambda x: x.price)
            offer_vars[slc] = pp.LpVariable.dicts(
                    name     = "offer_slc{}".format(slc),
                    indexs   = [o.idx for o in slc_supply],
                    lowBound = 0,
                    upBound  = 1
                )
            supply.append(slc_supply)

        profiles = dict()
        nbr_profiles = np.zeros(len(self.system.utilities)) #do we need this here?
        profile_vars = {}

        #collect all profiles
        for util in self.system.utilities:
            profiles[util.idx] = util.profilesForDay()
            nbr_profiles[util.idx] = len(profiles[util.idx])
            profile_vars[util.idx] = pp.LpVariable.dicts(
                    name    = "profile_util_{}".format(util.idx),
                    indexs  = profiles[util.idx],
                    cat     = "Binary"
                )

        flow_variables = {}
        for slc in range(_SLICES):
            variables = pp.LpVariable.dicts("edge_h{}".format(slc),edges)
            flow_variables[slc] = [variables[e] for e in edges]

        # construct social welfare description
        welfare = []
        # if _PROD_SURPLUS:
        #for generators
        for slc in range(_SLICES):
            for offer in supply[slc]:
                var = offer_vars[slc][offer.idx]
                welfare.append(-offer.price*offer.max*var)

        if _PROD_SURPLUS:
            # then for utilities / profile orders
            for util_idx,util_profs in profiles.items():
                for pid,profile in util_profs.items():
                    var = profile_vars[util_idx][pid]
                    welfare.append(np.sum(profile.profile*profile.price)*var)

        prob = pp.LpProblem("market",pp.LpMaximize)
        prob += pp.lpSum(welfare), "social welfare"

        # construct hourly balance constraints
        for slc in range(_SLICES):
            balance_gen = []
            for offer in supply[slc]:
                var = offer_vars[slc][offer.idx]
                balance_gen.append(offer.max*var)

            balance_demand = []
            for util_idx,util_profs in profiles.items():
                for pid,profile in util_profs.items():
                    var = profile_vars[util_idx][pid]
                    balance_demand.append(profile.profile[slc]*var)

            prob += pp.lpSum(balance_gen) == pp.lpSum(balance_demand), "balance for slc {}".format(slc)

        #construct cyclic and nodal constraints
        
        for slc in range(_SLICES):
            # cycles
            for c_idx in range(len(cycles)):
                eq = [v*c for v,c in zip(flow_variables[slc],cycle_matrix[:,c_idx])]
                prob += pp.lpSum(eq) == 0, "cycle flow {} for hour {}".format(c_idx,slc)

            for n_idx,node in enumerate(nodes):
        
                left = [v*d for d,v in zip(incidence_matrix[n_idx,:],flow_variables[slc])]
                
                right = []
                for offer in supply[slc]:
                    if offer.node == node:
                        var = offer_vars[slc][offer.idx]
                        right.append(offer.max*var)

                for util_idx,util_profs in profiles.items():
                    if self.system.utilities[util_idx].node == node:
                        for pid,profile in util_profs.items():
                            var = profile_vars[util_idx][pid]
                            right.append(-profile.profile[slc]*var)

            prob += pp.lpSum(left) == pp.lpSum(right), "nodal balance for {} at h {}".format(node,slc)

        # add profile selection constraints
        for util_idx,variables in profile_vars.items():
            prob += pp.lpSum(variables.values()) == 1, "profile selection for {}".format(util_idx)

        # first solve to get system price, i.e. no flow limits
        prob.solve()
        logger.debug("MIP status no flow constraints: {},{}".format(pp.LpStatus[prob.status],prob.status))

        # calculate system price

        sys_price = np.zeros(_SLICES)
        
        for slc in range(_SLICES):
            for offer in supply[slc]:
                var = offer_vars[slc][offer.idx]e
                if var.value() > 0:
                    sys_price[slc] = max(sys_price[slc],offer.price)

        # add flow limits

        for slc in range(_SLICES):
            for idx,(i,j) in enumerate(edges):
                if 'limit' in self.system.network.edges[i,j]:
                    limit = self.system.network.edges[i,j]['limit']
                    prob += flow_variables[slc][idx] <= limit, "limit of {} on line ({},{}) at h{}".format(limit,i,j,slc)
        
        # solve again this time with the flow limits, so we get possible price splits
        prob.solve()

        logger.debug("MIP status: {},{}".format(pp.LpStatus[prob.status],prob.status))
        logger.info("MIP welfare: {}".format(pp.value(prob.objective)))
        if prob.status < 0:
            raise ValueError("Could not solve MIP problem!")

        # save power flows
        for slc in range(_SLICES):
            for idx,(i,j) in enumerate(edges):
                flow = flow_variables[slc][idx].value()
                if "flow" in self.system.network[i][j]:
                    self.system.network[i][j]["flow"].append(flow)
                else:
                    self.system.network[i][j]["flow"] = [flow]


        selectedProfiles = {}
        profile = np.zeros(_SLICES)
        
        for util_idx,variables in profile_vars.items():
            # this can be done with list comprahensions but 
            # I think this is more clear to read without much speed penalty
            sel = 0
            for idx,var in variables.items():
                if var.value() > 0:
                    # there is only one var > 0
                    sel = idx
                    # so we can skip the rest
                    break
            selectedProfiles[util_idx] = profiles[util_idx][sel]
            profile += profiles[util_idx][sel].profile

        
        price = {}
        for z in self.system.zones_dict.values():
            price[z] = sys_price.copy()
        accepted = []
        
        for slc in range(_SLICES):
            accepted_slc = []
            for offer in supply[slc]:
                var = offer_vars[slc][offer.idx]
                if var.value() > 0:
                    item = ScheduleItem(offer.max*var.value(), offer.min, 
                        offer.max, offer.price, offer.idx, offer.t)
                    accepted_slc.append(item)
                    price[offer.node][slc] = max(price[offer.node][slc],offer.price)
            accepted.append(accepted_slc)
        price["sys"] = sys_price

        return selectedProfiles, profile, accepted, price
    


    def matchDemandForT(self, demand, t, slc):

        supply = [p.spotOfferForT(t) for p in self.system.producers]
        supply.sort(key=lambda x: x.price)  # sort inplace with first element
        matched = 0
        mismatch = 0
        mismatchedItem = None
        accepted = []
        price = None
        needsChange = False

        while matched < demand:
            offer = supply.pop(0)

            power = offer.max * SPOT_MARKET_RESOLUTION / 60  # to MWm to MWh

            if power > demand-matched:
                power = np.ceil(demand-matched)  # we take the next full MW

            matched += power
            acc = power / SPOT_MARKET_RESOLUTION * 60  # to MW

            if acc < offer.min and len(accepted) == 0:
                matched -= power
                continue

            if acc < offer.min:
                if len(accepted) == 0:
                    continue
                needsChange = True
                mismatch = offer.min - acc
                mismatchedItem = \
                    ScheduleItem(acc, offer.min, offer.max,
                                 offer.price, offer.idx, offer.t)
            else:
                item = \
                    ScheduleItem(acc, offer.min, offer.max,
                                 offer.price, offer.idx, offer.t)

                accepted.append(item)

        accepted.sort(key=lambda x: x.accepted-x.min, reverse=True)
        i = 0
        while mismatch > 0:
            item = accepted[i]
            scale = item.accepted - item.min

            if mismatch < scale:
                newAcc = item.accepted - mismatch
                mismatch = 0
            else:
                newAcc = item.min
                mismatch -= scale

            item.accepted = newAcc
            i += 1
            if i == len(accepted) and mismatch > 0:
                logger.debug(demand)
                logger.debug(accepted)
                logger.debug(supply)
                raise ValueError("supply and demand did not converge")
                # TODO implement rerun with different solution

        if mismatchedItem is not None:
            mismatchedItem.accepted = mismatchedItem.min
            accepted.append(mismatchedItem)
        # TODO: add mismatchedItem to accepted offers

        price = max([x.price for x in accepted])

        return price, accepted


class DemandAgent(Recorder):

    def __init__(self, system, maxD, minD, t0, isOptimizer=None, isRemote=None):
        super().__init__()

        self.system = system
        self.t0 = int(t0)
        self.t0_shift = 0
        self.maxD = maxD
        self.minD = minD
        self._demandForDay = None
        self.cost = 0
        self.usedPower = 0
        if isOptimizer is None:
            self._isOptimizer = np.random.rand() < OPTIMIZERS
        else:
            self._isOptimizer = isOptimizer
        if isRemote is None:
            self._isRemote = (np.random.rand() < REMOTE_CONTROLLED) \
                and self.isOptimizer
        else:
            self._isRemote = isRemote
        self.util_idx = None
        self.isCacheAgent = False
        self.nrOfCachedUsers = None

        self.addVariable('_usage')

        self.profileID = None
        self.node = 0

    @property
    def isOptimizer(self):
        if self._isOptimizer is None:
            self._isOptimizer = np.random.rand() < OPTIMIZERS
        return self._isOptimizer

    @property
    def isRemote(self):
        if self._isRemote is None:
            self._isRemote = (np.random.rand() < REMOTE_CONTROLLED) \
                and self.isOptimizer
        return self._isRemote

    def updatePostMarket(self):

        if self.isOptimizer:
            self.optimizeUsage()
        else:
            self.t0_shift = np.round(np.random.normal(0, 15))
        pass

    #@jit
    def demandForT(self, t, t0=None):
        """t can be array"""

        if t0 is None:
            t0 = self.t0 + self.t0_shift

        ret = calcDemandCython(t, t0, self.maxD, self.minD)

        return ret

    def demandForH(self, H):
        t = HtoMin(H)
        return np.sum(self.demandForT(t))/60.0  # we devide by 60 to get kWh

    @property
    def demandForDay(self):
        if self._demandForDay is None:
            D = self.demandForT(np.arange(LENGTH_OF_DAY))
            self._demandForDay = integrateDayToSlice(D)/60  # KWm to kWh
        return self._demandForDay

    def profilesForDay(self):
        if self.isOptimizer and self.isRemote:
            profiles = []
            # for slc in range(_SLICES):

            #   D = self.demandForT(np.arange(LENGTH_OF_DAY),
            #                       t0=slc * SPOT_MARKET_RESOLUTION)
            #   # profile = integrateDayToSlice(D)/60  # KWm to kWh
            #   profile = D
            #   profiles.append(profile)

            profiles = calcProfiles(np.arange(LENGTH_OF_DAY), _SLICES,
                                    SPOT_MARKET_RESOLUTION, self.maxD, self.minD)
            return profiles
        else:
            raise ValueError("agent has no profiles")

    def setProfileForDay(self, profileID):
        self.profileID = profileID

    def caculateRealTimeCosts(self, price):
        # we roud to the first decimal
        # from kWh to MWh for market prices
        p = np.round(self.demandForDay, 1)/1000.0
        self.usedPower += np.sum(self.demandForDay)
        if self.isCacheAgent:
            logger.debug("user from {} usage:{}".format(self.util_idx, p))
            logger.debug("user from {} cost:{}".format(self.util_idx, p*price))
            logger.debug("sum: {}".format(np.sum(np.round(p*price, 2))))
        p *= price

        return np.sum(np.round(p, 2))

    def optimizeUsage(self):

        if self.isRemote:
            self.t0 = self.profileID * SPOT_MARKET_RESOLUTION
        else:
            spotPrice = self.system.spotPrice[self.node]

            spotPrice = np.repeat(spotPrice, SPOT_MARKET_RESOLUTION)
            usage = self.demandForT(np.arange(LENGTH_OF_DAY))

            sig = abs(np.fft.ifft(np.fft.fft(spotPrice)
                                  * np.conj(np.fft.fft(usage))))
            if self.isCacheAgent:
                self.t0 += np.argmin(sig)
            else:
                self.t0 += np.argmin(sig) + np.round(np.random.normal(0, 15))
            self.t0 = int(self.t0 % LENGTH_OF_DAY)

    def nextDay(self):
        if self.system.daysRun > 0:
            price = self.system.spotPrice[self.node]

            rtc = self.caculateRealTimeCosts(price)
            self.cost += rtc
        if LOG_DEMAND:
            self._usage = self.demandForT(np.arange(LENGTH_OF_DAY))
        self._demandForDay = None

    def resetCosts(self):
        self.cost = 0
        self.usedPower = 0
        if LOG_DEMAND:
            self.resetHistoryOf('_usage')

class SimpleAppliance(DemandAgent):
    def __init__(self, system, maxD, minD, t0, t_ON, isOptimizer=None, isRemote=None):
        super().__init__(system, maxD, minD, t0, isOptimizer, isRemote)
        self.t_ON = int(t_ON)

    def demandForT(self, t, t0=None):
        """t can be array"""

        if t0 is None:
            t0 = self.t0 + int(self.t0_shift)

        ret = np.ones(LENGTH_OF_DAY) * self.minD


        if t0+self.t_ON > LENGTH_OF_DAY:
            t0 += LENGTH_OF_DAY - (t0 + self.t_ON)
        ret[t0:t0 + self.t_ON] = self.maxD

        
        return ret
    
    def profilesForDay(self):
        if self.isOptimizer and self.isRemote:
            profiles = []
            for slc in range(_SLICES):

              D = self.demandForT(np.arange(LENGTH_OF_DAY),
                                  t0=slc * SPOT_MARKET_RESOLUTION)
              # profile = integrateDayToSlice(D)/60  # KWm to kWh
              profile = D
              profiles.append(profile)

            
            return profiles
        else:
            raise ValueError("agent has no profiles")

class TwinPeaksDemandAgent(DemandAgent):
    def __init__(self, system, maxD, minD):
        super().__init__(system, maxD, minD, t0=0, isOptimizer=False, isRemote=False)

    @property
    def isOptimizer(self):
        return False

    @property
    def isRemote(self):
        return False

    def demandForT(self, t, t0=None):
        ret = np.ones(LENGTH_OF_DAY) * self.minD
        ret[0:12*60] += gaussian(12*60,150)*0.5*(self.maxD-self.minD)
        ret[10*60:] += gaussian(14*60,100)*(self.maxD-self.minD)
        ret[:] -= gaussian(LENGTH_OF_DAY,250)*0.05*self.minD

        return ret


class UtilityAgent(Recorder):

    def __init__(self, system, users, idx):
        super().__init__()

        self.addVariables(['balancingCost', 'purchaseCost'], 0)

        self.system = system
        self.users = users
        self._dailyDemand = None
        self._minuteDemand = None

        self._dailyForecast = None
        self.idx = idx
        self._demandHistory = []
        self._profileID = None
        self._profile = None
        for u in self.users:
            u.util_idx = idx

        self.cacheAgents = None
        self._baseloadForecast = None
        self._minuteProfiles = []

        maxNorm = 0
        maxFlex = 0
        for u in self.users:
            if u.isOptimizer:
                maxFlex += u.maxD
            else:
                maxNorm += u.maxD

        self.maxLoad = {"maxNorm": maxNorm, "maxFlex": maxFlex}
        self.node = 0

    def error(self):
        if ERROR_SCALE == 0:
            return 1
        else:
            return randomWalk(LENGTH_OF_DAY, 1, 0.00005, 0.00007)

    def minuteDemandSplit(self, current=False):
        t = np.arange(LENGTH_OF_DAY)
        if USE_CACHE_AGENTS:

            base = self.cacheAgents.normal.demandForT(t)
            if self.cacheAgents.optimizing.isRemote:
                remote = self.cacheAgents.optimizing.demandForT(t)
            else:
                base += self.cacheAgents.optimizing.demandForT(t)
                remote = np.zeros(LENGTH_OF_DAY)

            return (base / 1000.0, remote / 1000)  # from kWm to MWm

        if self._minuteDemand is None or current:
            base = np.zeros(LENGTH_OF_DAY)
            remote = np.zeros(LENGTH_OF_DAY)

            for u in self.users:
                if u.isOptimizer and u.isRemote:
                    remote += u.demandForT(t)
                else:
                    base += u.demandForT(t)
            self._minuteDemand = (
                base / 1000.0, remote / 1000)  # from kWm to MWm
        return self._minuteDemand

    @property
    def dailyDemand(self):
        if self._dailyDemand is None:
            D = integrateDayH(self.minuteDemand()) / 60  # from MWm to MWh
            self._dailyDemand = D
        return self._dailyDemand

    @property
    def numberOfUsers(self):
        return len(self.users)

    def baseloadForecast(self):
        if self._baseloadForecast is None:
            D = self.historicDemand()
            error = self.error()
            self._baseloadForecast = D*error
        return self._baseloadForecast

    def forecastDailyDemand(self):
        return integrateDayToSlice(self._dailyForecast/60)   # from MWm to MWh

    def revenues(self):
        if USE_CACHE_AGENTS:
            rev = sum([u.cost for u in self.cacheAgents])
        else:
            rev = sum([u.cost for u in self.users])
        return rev

    def costs(self):
        ret = dict()
        fixed = self.fixedCosts()
        if USE_CACHE_AGENTS:
            ret['normal_variable'] = self.cacheAgents.normal.cost
             
            ret['opt_variable'] = self.cacheAgents.optimizing.cost

            ret['all'] = self.cacheAgents.normal.cost
            ret['all'] += self.cacheAgents.optimizing.cost
            
            ret['fixed'] = fixed

        else:
            opt = 0
            normal = 0
            for u in self.users:
                if u.isOptimizer:
                    opt+= u.cost
                else:
                    normal += u.cost

            if normal > 0:
                ret['normal_variable'] = normal
            else:
                ret['normal_variable'] = 0
            if opt > 0:
                ret['opt_variable'] = opt
            else:
                ret['opt_variable'] = 0
            ret['all'] = normal + opt
            ret['fixed'] = fixed

        return ret

    def costPerUser(self):
        ret = dict()
        fixed = self.fixedCostPerUser()
        if USE_CACHE_AGENTS:
            ret['normal'] = self.cacheAgents.normal.cost
            ret['normal'] /= self.cacheAgents.normal.nrOfCachedUsers
            ret['normal_variable'] = ret['normal']
            ret['normal'] += fixed

            ret['optimizing'] = self.cacheAgents.optimizing.cost
            ret['optimizing'] /= self.cacheAgents.optimizing.nrOfCachedUsers
            ret['opt_variable'] = ret['optimizing']
            ret['optimizing'] += fixed

            ret['all'] = self.cacheAgents.normal.cost
            ret['all'] += self.cacheAgents.optimizing.cost
            ret['all'] /= len(self.users)
            ret['all'] += fixed

        else:
            opt = []
            normal = []
            for u in self.users:
                if u.isOptimizer:
                    opt.append(u.cost)
                else:
                    normal.append(u.cost)

            if len(normal) > 0:
                ret['normal'] = np.mean(normal) + fixed
                ret['normal_variable'] = np.mean(normal)
            else:
                ret['normal'] = 0
                ret['normal_variable'] = 0
            if len(opt) > 0:
                ret['optimizing'] = np.mean(opt) + fixed
                ret['opt_variable'] = np.mean(opt)
            else:
                ret['optimizing'] = 0
                ret['opt_variable'] = 0
            normal.extend(opt)
            ret['all'] = np.mean(normal) + fixed

        return ret

    def profilesForDay(self):

        base_profile = self.baseloadForecast()  # in MWm

        if USE_CACHE_AGENTS:
            if self.cacheAgents.optimizing.isRemote:
                variable_profiles = self.cacheAgents.optimizing.profilesForDay()
            else:
                variable_profiles = np.zeros((_SLICES, LENGTH_OF_DAY))
        else:
            variable_profiles = np.zeros((_SLICES, LENGTH_OF_DAY))
            for u in self.users:
                # TODO: loosen the assumption that optimizers have _SLICES
                # profiles

                if u.isOptimizer and u.isRemote:

                    profiles = u.profilesForDay()

                    variable_profiles += profiles

        # for i in range(_SLICES):
        #   variable_profiles[i,:] *= randomWalk(LENGTH_OF_DAY,1,0.0001,0.0001)
        variable_profiles /= 1000  # kWm to MWm
        variable_profiles *= self.error()

        # numpy magic adds base to every entry
        sum_profiles = variable_profiles + base_profile

        base_slices = integrateDayToSlice(base_profile)/60  # MWm to MWh

        self._minuteProfiles = sum_profiles

        profiles = {}

        # test if sum_profiles have sufficently high difference
        test = np.sum(np.abs(sum_profiles[1:, :] - sum_profiles[0, :]), axis=1)
        test = np.concatenate(([True], test > 1))

        for idx, p in enumerate(sum_profiles):
            D = integrateDayToSlice(p)/60  # MWm to MWh
            if test[idx]:
                profiles[idx] = Profile(D, 3000*np.ones(_SLICES), self.idx, idx,
                                        base_slices)

        return profiles

    def setProfileForDay(self, profile):
        self._profile = profile
        self._profileID = profile.id
        logger.info("profile set to {} for {}".format(
            self._profileID, self.idx))
        if USE_CACHE_AGENTS:
            self.cacheAgents.optimizing.setProfileForDay(self._profileID)
        else:
            for u in self.users:
                if u.isOptimizer:
                    u.setProfileForDay(self._profileID)
        self._dailyForecast = self._minuteProfiles[self._profileID]

    def fixedCostPerUser(self):
        fC = -(self.revenues() - self.balancingCost -
               self.purchaseCost)/self.numberOfUsers
        return fC

    def fixedCosts(self):
        fC = -(self.revenues() - self.balancingCost - self.purchaseCost)
        return fC

    def updateDemandHistory(self):
        self._demandHistory.append(self.minuteDemandSplit()[
                                   0])  
        # we only follow base load as for the profiles we don‘t 
        # need to make any forecasts. Baseload also includes 
        # flexible users

    def updatePostMarket(self):
        if USE_CACHE_AGENTS:
            # not sure weather to include this extra randomization
            # self.cacheAgents.normal.updatePostMarket()
            self.cacheAgents.optimizing.updatePostMarket()
        else:
            for u in self.users:
                u.updatePostMarket()

    def historicDemand(self, c=None, l=30):
        if c is None:
            c = UTILITY_DEMAND_DECAY

        if len(self._demandHistory) > 1:
            x = np.array(self._demandHistory[-l:])
            w = (np.ones(x.shape)-c) ** (np.repeat(
                np.arange(x.shape[0]-1, -1, -1), x.shape[1]
            )
                .reshape(x.shape))
            w /= np.sum(w, axis=0)
            forecast = np.sum(x*w, axis=0)
            return forecast
        else:
            return self.minuteDemandSplit()[0]

    def setupCacheAgents(self):
        demand_normal = np.zeros(LENGTH_OF_DAY)
        demand_optimizers = np.zeros(LENGTH_OF_DAY)
        t = np.arange(LENGTH_OF_DAY)
        t_0_opt = []
        t_0_norm = []
        for u in self.users:
            if u.isOptimizer:
                demand_optimizers += u.demandForT(t)
                t_0_opt.append(u.t0)
            else:
                demand_normal += u.demandForT(t)
                t_0_norm.append(u.t0)

        if len(t_0_opt) == 0:
            t_0_opt = [0]
        if len(t_0_norm) == 0:
            t_0_norm = [0]

        normalAgent = DemandAgent(self.system,
                                  np.max(demand_normal),
                                  np.min(demand_normal),
                                  np.mean(t_0_norm),
                                  False)
        normalAgent.util_idx = self.idx

        logger.debug("normalAgent max: {}, min: {} t0: {}".format(
            normalAgent.maxD/1000,
            normalAgent.minD/1000,
            normalAgent.t0))

        optimizingAgent = DemandAgent(self.system,
                                      np.max(demand_optimizers),
                                      np.min(demand_optimizers),
                                      np.mean(t_0_opt),
                                      True)
        optimizingAgent.util_idx = self.idx

        logger.debug("optimizingAgent max: {}, min: {} t0: {}".format(
            optimizingAgent.maxD/1000,
            optimizingAgent.minD/1000,
            optimizingAgent.t0))

        normalAgent.isCacheAgent = True
        normalAgent.nrOfCachedUsers = len(t_0_norm)
        optimizingAgent.isCacheAgent = True
        optimizingAgent.nrOfCachedUsers = len(t_0_opt)

        self.cacheAgents = AgentCache(normalAgent, optimizingAgent)

    def startDay(self):
        if USE_CACHE_AGENTS and self.cacheAgents is None:
            self.setupCacheAgents()

    def nextDay(self):

        if USE_CACHE_AGENTS and self.cacheAgents is None:
            self.setupCacheAgents()

        if self.system.daysRun > 0:
            self.updateDemandHistory()

        self._dailyDemand = None
        self._minuteDemand = None
        self._dailyForecast = None
        self._profileID = None
        self._profile = None
        self._baseloadForecast = None
        self._minuteProfiles = []

        if USE_CACHE_AGENTS:
            self.cacheAgents.normal.nextDay()
            self.cacheAgents.optimizing.nextDay()
        else:
            for u in self.users:
                u.nextDay()

    def resetCosts(self):
        self.resetHistoryOf('balancingCost', 0)
        self.resetHistoryOf('purchaseCost', 0)

        if USE_CACHE_AGENTS and self.cacheAgents is not None:
            for u in self.cacheAgents:
                u.resetCosts()
        else:
            for u in self.users:
                u.resetCosts()


class ProductionAgent(Recorder):

    def __init__(self, system, idx, capacity, price):
        super().__init__()
        self.addVariables(['revenues', 'balancingRev', 'cost'], 0)
        self.addVariables(['usedCapacity'])

        self.system = system
        self.maxCapacity = capacity
        self._capacity = np.ones(LENGTH_OF_DAY)*capacity

        self.addVariables(['priceMW'], price)
        self.costMW = price
        self.idx = idx
        self.initRegulationFactor(REGULATION_FACTOR)
        self.regulationPriceFactor = REGULATION_PRICE_FACTOR
        self.regulationUpdateFactor = 2
        #self.minCapacity = self.maxCapacity * MIN_RUN_FACTOR
        self.addVariable("_production")
        self._schedule = np.zeros(LENGTH_OF_DAY)
        self.partOfSpot = True

        self.error = 1
        self.minRunFactor = MIN_RUN_FACTOR
        self.daysRun = 0
        self.node = 0

        

    @property
    def capacity(self):
        return self._capacity

    def regulationOfferForT(self, t):
        #RegulationOffer = recordclass("RegulationOffer","up down priceUp priceDown idx t")

        up = self.upRegCapacityForT(t)
        if up < BALANCING_MIN_OFFER:
            up = 0
        down = self.downRegCapacityForT(t)
        if down < BALANCING_MIN_OFFER:
            down = 0
        if down == 0 and up == 0:
            return None

        priceUp = self.regPriceForUpT(t)
        priceDown = self.regPriceForDownT(t)

        offer = RegulationOffer(up, down, priceUp, priceDown, self.idx, t)



        return offer

    def priceForT(self, t):
        return self.priceMW

    def regPriceForUpT(self, t):
        return np.round(self.priceForT(t) * (1+self.regulationPriceFactor), 2)

    def regPriceForDownT(self, t):
        return np.round(self.priceForT(t) * (1-self.regulationPriceFactor), 2)

    def upRegCapacityForT(self, t):
        if np.all(self._schedule[t] >= self._capacity[t]*self.minRunFactor):
            up = np.floor(np.min(self.capacity[t] - self._schedule[t]))
            up *= self.regulationFactor
        else:
            up = 0

        return np.floor(up)

    def upCapacityForT(self, t):
        up = np.mean(self.capacity[t] - self._schedule[t])
        return np.floor(up)

    def downRegCapacityForT(self, t):
        down = np.min(self._schedule[t] - self.capacity[t]*self.minRunFactor)
        down *= self.regulationFactor

        return np.floor(down)

    def spotOfferForT(self, t):
        if self.partOfSpot:
            up = self.upCapacityForT(t) * MAX_SPOT_BID
            return Offer(up, self.minRunFactor*up, self.priceForT(t), self.idx, t, self.node)
        else:
            return Offer(0, 0, 0, self.idx, t, self.node)

    def upRegOfferForT(self, t):
        up = self.upRegCapacityForT(t)
        return Offer(up, 0, self.regPriceForUpT(t), self.idx, t, self.node)

    def downRegOfferForT(self, t):
        down = self.downRegCapacityForT(t)

        return Offer(down, 0, self.regPriceForDownT(t), self.idx, t, self.node)

    def setMinuteSchedule(self, schedule):

        assert len(schedule) == LENGTH_OF_DAY
        self._schedule = schedule.copy()

    def setSchedule(self, schedule):
        assert len(schedule) == _SLICES
        for s in range(_SLICES):
            t = slice2T(s)
            self._schedule[t] = schedule[s]*np.ones(SPOT_MARKET_RESOLUTION)

    def minuteSchedule(self):
        return self._schedule

    def minuteProduction(self):
        return self._schedule * self.error

    def initRegulationFactor(self, reg):
        self._regulationFactor = reg
        self.regulationFactor = reg

    def applyBlcCorrectionForT(self, t, value):
        self.updateScheduleForT(t, value)

        # we decrease the amount of regulation available for the next BlcSlc
        # while increasing the price, once we need to change schedule
        self.regulationFactor /= self.regulationUpdateFactor
        self.regulationPriceFactor *= self.regulationUpdateFactor

    def updateScheduleForT(self, t, value):

        if value > 0 and \
           value > np.min(self.capacity[t] - self._schedule[t]):
            raise ValueError("requested capacity ({}) not available({})."
                             .format(value, np.min(self.capacity[t] - self._schedule[t])))

        if (value < 0) and \
                (abs(value) > np.min(self._schedule[t])):
            raise ValueError("requested reduction ({}) not possible({})."
                             .format(value, self._schedule[t] - self.capacity[t]*self.minRunFactor))

        # print "update {} from {} to {}".format(
        # self.idx,self._schedule[t[0]],self._schedule[t][0] + value)
        self._schedule[t] += value
        self.system.updateProductionScheduleForT(t, value)

    def updateScheduleForHq(self, H, q, value):
        t = HqToMin(H, q)
        self.updateScheduleForT(t, value)

    def calculateCost(self):
        C = np.sum(self.minuteProduction())/60 * self.costMW
        self.cost += C
        return C

    def calculateProfit(self):
        return self.revenues \
            + self.balancingRev \
            - self.cost

    def nextDay(self):
        self.calculateCost()
        self._production = self.minuteProduction()
        self.usedCapacity = np.sum(
            self._schedule*self.error)/np.sum(self._capacity)
        self._schedule = np.zeros(LENGTH_OF_DAY)
        self.daysRun += 1

    def nextSlc(self):
        self.regulationPriceFactor = REGULATION_PRICE_FACTOR
        self.regulationFactor = self._regulationFactor

    def isRenewable(self):
        return False

    @property
    def type(self):
        return "convetional"

    def resetCosts(self):
        self.resetHistoryOf('revenues', 0)
        self.resetHistoryOf('balancingRev', 0)
        self.resetHistoryOf('cost', 0)
        self.resetHistoryOf('usedCapacity', 0)
        self.resetHistoryOf('_production')
        self.daysRun = 0


class RenewableAgent(ProductionAgent):

    def __init__(self, system, idx, capacity, price):
        super().__init__(system, idx, capacity, price)
        self.minRunFactor = 0
        # self.K = np.arange(5,100,5)
        # self.r = 0.5
        # self.e = 0.5
        # self.q = np.ones(self.K.shape)*500
        # self.j = 0
        self.costMW = 4
        self.addVariable('_scheduleLog')
        self.addVariable('_capacityLog')
        # roughtly what a windfarm needs to earn per MWh to payback the
        # investment costs

    def upCapacityForT(self, t):
        return super().upCapacityForT(t) * RENEWABLE_SAFETY_MARGIN

    def upRegCapacityForT(self, t):
        return super().upRegCapacityForT(t) * RENEWABLE_REG_FACTOR

    def downRegCapacityForT(self, t):
        return super().downRegCapacityForT(t) * RENEWABLE_REG_FACTOR

    def isRenewable(self):
        return True

    def calculateCost(self):
        return 0

    def resetCosts(self):
        super().resetCosts()
        self.resetHistoryOf('_scheduleLog')
        self.resetHistoryOf('_capacityLog')

    def nextDay(self):
        self._scheduleLog = self._schedule.copy()
        self._capacityLog = self._capacity.copy()
        super().nextDay()


    @ProductionAgent.type.getter
    def type(self):
        return "renewable"

    def minuteProduction(self):
        return self._capacity * self.error


class SolarAgent(RenewableAgent):

    def __init__(self, system, idx, capacity, price):
        super().__init__(system, idx, capacity, price)
        self.maxCapacity = capacity
        self._capacity = self._forecastCapacity()
        # resembels the general shape for a solar pannel
        # determins how much cloud coverage is present
        #self.error = 1 + np.random.normal(0,SOLAR_ERROR,LENGTH_OF_DAY)
        self.error = system.renewableError()

    def _forecastCapacity(self):
        capacity = self.maxCapacity \
            * np.roll(gaussian(LENGTH_OF_DAY, 2*60),
                      self.system.solarTime + np.random.randint(15))\
            * (normRandomWalk() + RENEWABLE_CORRELATION
               * self.system._renewableCorrelation)\
            / (1+RENEWABLE_CORRELATION)

        return capacity

    def nextDay(self):
        super().nextDay()
        self.error = self.system.renewableError()
        self._capacity = self._forecastCapacity()

    @ProductionAgent.type.getter
    def type(self):
        return "solar"

# TODO: fix overhang of capacity due to shifting


class WindAgent(RenewableAgent):

    def __init__(self, system, idx, capacity, price):
        super().__init__(system, idx, capacity, price)
        self.maxCapacity = capacity
        self._capacity = self._rnd_capacity()

        self.error = system.renewableError()

    def _rnd_capacity(self):
        ret = np.zeros(LENGTH_OF_DAY)
        for i in range(1+np.random.randint(5)):
            ret += np.roll(gaussian(LENGTH_OF_DAY, 2*60),
                           np.random.randint(LENGTH_OF_DAY))

        ret /= np.max(ret)
        ret *= normRandomWalk()
        ret *= np.random.rand()
        return ret * self.maxCapacity

    def nextDay(self):
        super().nextDay()
        self.error = self.system.renewableError()
        self._capacity = self._rnd_capacity()

    @ProductionAgent.type.getter
    def type(self):
        return "wind"


def normRandomWalk(a=0.01, b=0.01):
    w = randomWalk(LENGTH_OF_DAY, 1, 0.01, 0.01)
    w -= np.max(w) - 1
    return w


