function [Call, Put] = CRRPricing(S0, X, r, T, sigma, steps)


    StartDate = '11-Jan-2000';
    date_number = datenum(StartDate, 'dd-mmm-yyyy');
    date_number = addtodate(date_number, round(T*12), 'month');
    EndDate = datestr(date_number, 'dd-mmm-yyyy');
    StockSpec = stockspec(sigma, S0);
    RateSpec = intenvset('Rates', r, 'StartDates', StartDate, 'EndDates', EndDate, 'Compounding', 1);
    TimeSpec = crrtimespec(StartDate, EndDate, steps);
    
    CRRTree = crrtree(StockSpec, RateSpec, TimeSpec);
    
    [Call, ~] = optstockbycrr(CRRTree, 'Call', X, StartDate, EndDate);
    [Put, ~] = optstockbycrr(CRRTree, 'Put', X, StartDate, EndDate);
end