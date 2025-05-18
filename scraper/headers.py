def get_headers(date, page, data_only=False):
    url = 'https://merolagani.com/Floorsheet.aspx'
    cookies = {
        'ASP.NET_SessionId': 'myrsoi4k1ac3qi4hpgfb0vrp',
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:134.0) Gecko/20100101 Firefox/134.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Origin': 'https://merolagani.com',
        'Connection': 'keep-alive',
        'Referer': 'https://merolagani.com/Floorsheet.aspx',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-User': '?1',
        'Priority': 'u=0, i',
    }

    data = {
        '__EVENTTARGET': '',
        '__EVENTARGUMENT': '',
        '__VIEWSTATE': 'xGoWUhJs6ga452ou5ZhFURPiTrOn1kuAl9wYj03FPTenyBHXFhlwmVCqznsAUyWhZkEh8M/yUQX3Agavnx7xgm2EXPugxIkMqzwtoYgNMTX/JFcQBg326X9UEIQYZtizRzaaIuzRHsfd4w/IgIIxM+U8wSMJvefq4dzYPJ5FZGM8KrA8hxpIV2w1TcKa4PbEFvOgPJgCPEUFZLjjSZqE5e+CCKVgLd9k/X+2QtFG8X0uUYKWktH47vyQqXiAR8bfNt+9/ORyrIX/zU4Z8wP74Im7cK3ae2/wDmiGyFHkHzuxr+M3oYF+TJ/rqjxvcuswXFN3tZJ2Nrz+8yJ6j/Ghl5ofhLfFIBDtr9fjfFt0sapVo6YRM+OoIIMP90M3E41qYBKKgFOHlRsKFoxmjzk+/6bhjiup1r7z3RQZ+B91TZ+/LDOxwfVYeDIQtC6TW31tjokvfdB5ag3kDtmY/9CZFzLjSFVcW6JO+OGZeMb2xkPUSoerJV6tY1aXTvYfW0czOKeeJC1F7m6SDo5wzq2gsGVcIGXLzv3U0zXfz2iegVp2nnWlmgPYJLKcQ5a3EfFzEIIIokvdSASkG8PA8x/8qvax59xOGOdefLFTIVYOrlS33OTsD/mvEt8kpWJLpltFqF142zOfeHAxiQ6CaCiOYnzy16eGppwdRwsHj97Qn57vnWYd/r/EnKJlknUFYQKPHHAsHbybCRoKpAZ+IUL2LTzPrOFTdJzKAx1FIMqRGhIlBlkTwHevI4YsOia1TM4ImNENjBdTbHFD9jLdQBYI/rFJkXG6Yi4h1greLaoittNWLc0ojAIfk+RATPI2nWtnd3MRtlsnAvgJlMl6w/x/9cVWbP+FIl26hG+AyE54PfvT+Q5mVvh/bCqOAcw25RKYtvzicZbn8OMYQt987UfqAKjztuBmj0/6dO13C6LbIEGqgOXr1FSJE+IudwSPFtfRyehxKdFrVam+daf/zTFVYExomGSE691qwqsIsh145k1oMI7UHTayosZ4jaVmN2Uy',
        '__VIEWSTATEGENERATOR': '1F15F17F',
        '__EVENTVALIDATION': 'oPmDYHlfQ+5Tqd5SAyiLAGmJ9genMlbv9zmIfOocj3zKQSEBXjqqIftJo+4UBh6NaNIIQI+gskJBmCdhSii4r/i4Qvz+dO0I0WHXaEq+9u1Hmb8Pc27Jznn0lHAQ+rSKDh0lc1tRGXJdqcwgqXphV+jWsY2VmKUyB8ZV89ch121rNy4EuDPm/jxErilGp2SVTjWiHhz7/HsHgySjIkNiASo1JbLSp59sG/01MSmR2eHx5I+0/ARuLxwwhiOudJLNOrisJV3LPYBOAWDv31lJVsbDYV9TSaDkM0uLHkfGlIFeupoSJS1EZrBsOLN4PRVX/1njn1SMAbwljzSu6EP0bLjwtj/XT9izXzBBaRlyAtFCvwl8qon/BePCgXviY6l9dA2QjpD6l2cmBHCAhp9Yl2s1uSRaNW99oAKsGishM3dqxDKGOsvaBLbRaq1PWA6z3FeIbVofjLMHSwr6ZhsoKMX+wJoXV5o2d0vCkwsaUi/16xkLvWHmUEDMmE3kmXmdCflVOktxbEJEQy28pxncPlTQA5rHsHL/DlxSg3Uoe0DEbACM/8KFsNl0PjuOpohQ',
        'ctl00$ASCompany$hdnAutoSuggest': '0',
        'ctl00$ASCompany$txtAutoSuggest': '',
        'ctl00$txtNews': '',
        'ctl00$AutoSuggest1$hdnAutoSuggest': '0',
        'ctl00$AutoSuggest1$txtAutoSuggest': '',
        'ctl00$ContentPlaceHolder1$ASCompanyFilter$hdnAutoSuggest': '0',
        'ctl00$ContentPlaceHolder1$ASCompanyFilter$txtAutoSuggest': '',
        'ctl00$ContentPlaceHolder1$txtBuyerBrokerCodeFilter': '',
        'ctl00$ContentPlaceHolder1$txtSellerBrokerCodeFilter': '',
        'ctl00$ContentPlaceHolder1$txtFloorsheetDateFilter': date,
        'ctl00$ContentPlaceHolder1$PagerControl1$hdnPCID': 'PC1',
        'ctl00$ContentPlaceHolder1$PagerControl1$hdnCurrentPage': page,
        'ctl00$ContentPlaceHolder1$PagerControl1$btnPaging': '',
        'ctl00$ContentPlaceHolder1$PagerControl2$hdnPCID': 'PC2',
        'ctl00$ContentPlaceHolder1$PagerControl2$hdnCurrentPage': '0',
    }

    if data_only:
        return data
    
    return url, headers, cookies, data

def get_header_for_patro():
    """
    Get the header for Patro website
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:134.0) Gecko/20100101 Firefox/134.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Origin': 'https://www.hamropatro.com',
        'Connection': 'keep-alive',
        'Referer': 'https://www.hamropatro.com/calendar',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-User': '?1',
        'Priority': 'u=0, i',
    }
    
    return {'headers': headers}