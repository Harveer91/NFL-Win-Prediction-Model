import React, {useState}  from "react";
function Button(){
    const[isOpen, setIsOpen] = useState(false);
    const[selectedTeam, setSelectedTeam] = useState("Select a Team");
    
    const teams = ["Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
    "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
    "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
    "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Kansas City Chiefs",
    "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams", "Miami Dolphins",
    "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants",
    "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers", "San Francisco 49ers",
    "Seattle Seahawks", "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders"];
    
    const ToggleOpen = () =>{
        setIsOpen(!isOpen); //toggles the dropdown menu to open
    };

    const HandleSelect = (team) =>{
        setSelectedTeam(team); //upates the team that was selected and 
        setIsOpen(false);
    };

    return(
        <div className="dropdown">
            <button id="Team-selector" onClick={ToggleOpen}>{selectedTeam}</button>
            {isOpen && ( // use of conditional rendering to determine if if the button isOpen
                <ul className="dropdown-menu">
                    {teams.map((team,index) => (
                        <li id="option" key={index} onClick={() => HandleSelect(team)}>
                            {team}
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}
export default Button